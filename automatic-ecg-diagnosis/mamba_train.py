#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG (PTB-XL) classification with (Bi)Mamba in PyTorch.
- HDF5/CSV pipeline with official folds (1-8 train, 9 val, 10 test)
- Multi-label, logits head + Focal Loss
- AdamW + AMP + EarlyStopping on val_AUCPR
"""

import os, math, json, argparse, h5py, random, time
import numpy as np
import pandas as pd
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import average_precision_score, roc_auc_score


# ------------------------
# Utils: reproducibility
# ------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ------------------------
# Data
# ------------------------
class H5ECGDataset(Dataset):
    """
    Reads tracings from HDF5 and labels from CSV with explicit ecg_id alignment if available.
    - HDF5 dataset name defaults to 'tracings' with shape (N, 4096, 12)
    - Returns X: (12, 4096), Y: (C,)
    - Optional light augmentation for training
    """
    def __init__(self, h5_path: str, df: pd.DataFrame, label_cols: List[str],
                 dataset_name: str = "tracings", augment: bool = False):
        super().__init__()
        self.h5 = h5py.File(h5_path, "r")
        self.Xds = self.h5[dataset_name]
        self.Nh5 = self.Xds.shape[0]
        self.df = df.reset_index(drop=True).copy()
        self.label_cols = label_cols
        self.augment = augment

        # Build row index mapping
        if "ecg_id" in self.h5.keys() and "ecg_id" in self.df.columns:
            h5_ids = self.h5["ecg_id"][:].astype(int)
            id2idx = {int(e): int(i) for i, e in enumerate(h5_ids)}
            self.idx = np.asarray([id2idx[int(e)] for e in self.df["ecg_id"].astype(int).values], dtype=np.int64)
        else:
            if len(self.df) > self.Nh5:
                raise ValueError("CSV has more rows than HDF5; cannot align without ecg_id.")
            self.idx = np.arange(len(self.df), dtype=np.int64)

        self.Y = self.df[self.label_cols].astype(np.float32).values
        self.n_classes = self.Y.shape[1]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = int(self.idx[i])
        x = self.Xds[k, :, :]            # (4096, 12)
        x = np.asarray(x, dtype=np.float32)
        if self.augment:
            x = self._augment(x)
        # to torch: (C,L)
        x = torch.from_numpy(x).transpose(0, 1)  # (12, 4096)
        y = torch.from_numpy(self.Y[i])
        return x, y

    def _augment(self, x: np.ndarray):
        # x: (4096, 12)
        # amplitude scale ±10%
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale
        # gaussian noise
        x = x + np.random.normal(0, 0.003, size=x.shape).astype(np.float32)
        # shift ±0.2s @400Hz => ±80
        shift = np.random.randint(-80, 81)
        if shift != 0:
            x = np.roll(x, shift, axis=0)
        # short time mask
        if np.random.rand() < 0.3:
            L = x.shape[0]
            w = np.random.randint(int(0.01 * L), int(0.015 * L))
            s = np.random.randint(0, L - w)
            x[s:s + w, :] = 0.0
        return x

    def close(self):
        try:
            self.h5.close()
        except:
            pass

    def __del__(self):
        self.close()


def split_official(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trn = df[df["strat_fold"].isin([1, 2, 3, 4, 5, 6, 7, 8])].reset_index(drop=True)
    val = df[df["strat_fold"] == 9].reset_index(drop=True)
    tst = df[df["strat_fold"] == 10].reset_index(drop=True)
    return trn, val, tst


# ------------------------
# Loss: Focal (logits)
# ------------------------
class FocalLossLogits(nn.Module):
    """
    Multi-label focal loss (logits).
    alpha: tensor [C], per-class weight (e.g., alpha_c=clip(1-prevalence_c, 0.5, 0.95))
    gamma: float, focusing parameter
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: (B,C), targets: (B,C)
        prob = torch.sigmoid(logits)
        pt = targets * prob + (1 - targets) * (1 - prob)
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # focal term
        focal = (1.0 - pt).clamp_min(1e-7).pow(self.gamma)
        # BCE (prob form with logits is fine; we work in prob space here)
        bce = - (targets * torch.log(prob.clamp_min(1e-7)) +
                 (1 - targets) * torch.log((1 - prob).clamp_min(1e-7)))
        loss = w * focal * bce
        return loss.mean()


def alpha_from_prevalence(prev: np.ndarray, lo=0.5, hi=0.95) -> np.ndarray:
    return np.clip(1.0 - prev, lo, hi).astype(np.float32)


# ------------------------
# Model: Conv stem + (Bi)Mamba blocks + GAP head
# ------------------------
def build_mamba_module(d_model: int):
    """
    Try best-effort to import Mamba from different entry points.
    """
    try:
        from mamba_ssm import Mamba  # common API
        return Mamba(d_model)
    except Exception:
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
            return Mamba2(d_model=d_model)
        except Exception as e:
            raise ImportError(
                f"Cannot import Mamba. Please install 'mamba-ssm' properly. Original error: {e}"
            )


class MambaBlock(nn.Module):
    """ PreNorm Mamba block with residual """
    def __init__(self, d_model: int, p_drop: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = build_mamba_module(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # x: (B, L, D)
        h = self.norm(x)
        h = self.mamba(h)       # (B, L, D)
        h = self.dropout(h)
        return x + h


class BiMambaBlock(nn.Module):
    """ Bidirectional Mamba: forward + reversed branch, concat + projection + residual """
    def __init__(self, d_model: int, p_drop: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = build_mamba_module(d_model)
        self.bwd = build_mamba_module(d_model)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # x: (B, L, D)
        h = self.norm(x)
        y_fwd = self.fwd(h)                     # (B,L,D)
        y_bwd = self.bwd(torch.flip(h, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])     # align back
        y = torch.cat([y_fwd, y_bwd], dim=-1)   # (B,L,2D)
        y = self.proj(y)
        y = self.dropout(y)
        return x + y


class ECGMambaNet(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 256, n_blocks: int = 3,
                 p_drop: float = 0.2, bimamba: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(BiMambaBlock(d_model, p_drop) if bimamba else MambaBlock(d_model, p_drop))
        self.backbone = nn.Sequential(*blocks)
        self.head_norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(d_model, n_classes)   # logits

    def forward(self, x):
        # x: (B, 12, 4096)
        h = self.stem(x)                  # (B, D, L)
        h = h.transpose(1, 2)             # (B, L, D) for Mamba
        h = self.backbone(h)              # (B, L, D)
        h = self.head_norm(h)
        h = h.mean(dim=1)                 # GAP over L -> (B, D)
        h = self.head_dropout(h)
        logits = self.classifier(h)       # (B, C)
        return logits


# ------------------------
# Metrics
# ------------------------
@torch.no_grad()
def evaluate(model, loader, device, n_classes: int):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        ys.append(yb.cpu().numpy())
        ps.append(prob)
    Y = np.concatenate(ys, axis=0)  # (N,C)
    P = np.concatenate(ps, axis=0)  # (N,C)

    metrics = {}
    # macro AUCPR / AUCROC（若全0或全1则返回nan）
    try:
        metrics["AUCPR_macro"] = float(average_precision_score(Y, P, average="macro"))
    except Exception:
        metrics["AUCPR_macro"] = float("nan")
    try:
        metrics["AUCROC_macro"] = float(roc_auc_score(Y, P, average="macro"))
    except Exception:
        metrics["AUCROC_macro"] = float("nan")

    # per-class
    per_cls = []
    for i in range(n_classes):
        y = Y[:, i].astype(int)
        p = P[:, i]
        prev = float(y.mean())
        ap = float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
        try:
            auc = float(roc_auc_score(y, p)) if (y.min() != y.max()) else float("nan")
        except Exception:
            auc = float("nan")
        per_cls.append({"prevalence": prev, "AUCPR": ap, "AUCROC": auc})
    metrics["per_class"] = per_cls
    return metrics


# ------------------------
# Training loop
# ------------------------
def train_loop(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(args.path_to_csv)
    if "strat_fold" not in df.columns:
        raise ValueError("CSV must include 'strat_fold' column.")
    label_cols = [c.strip() for c in args.labels.split(",") if c.strip()]

    df_trn, df_val, df_tst = split_official(df)
    Y_trn = df_trn[label_cols].astype(np.float32).values
    Y_val = df_val[label_cols].astype(np.float32).values
    Y_tst = df_tst[label_cols].astype(np.float32).values
    n_classes = Y_trn.shape[1]

    # alpha for focal
    if args.alpha_mode == "train":
        prev = Y_trn.mean(axis=0)
    elif args.alpha_mode == "val":
        prev = Y_val.mean(axis=0)
    elif args.alpha_mode == "const":
        prev = None
    elif args.alpha_mode == "file" and os.path.exists(args.alpha_file):
        with open(args.alpha_file, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            alpha_vec = np.asarray(data, dtype=np.float32)
        else:
            alpha_vec = np.asarray([float(data[c]) for c in label_cols], dtype=np.float32)
        prev = None
    else:
        prev = Y_trn.mean(axis=0)

    if prev is not None:
        alpha_vec = alpha_from_prevalence(prev, lo=0.5, hi=0.95)
    elif args.alpha_mode == "const":
        alpha_vec = np.full((n_classes,), float(args.alpha_const), dtype=np.float32)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "focal_alpha.json"), "w") as f:
        json.dump({label_cols[i]: float(alpha_vec[i]) for i in range(n_classes)}, f, indent=2)
    with open(os.path.join(args.save_dir, "label_order.json"), "w") as f:
        json.dump(label_cols, f, indent=2)

    # datasets / loaders
    ds_trn = H5ECGDataset(args.path_to_hdf5, df_trn, label_cols, dataset_name=args.dataset_name, augment=True)
    ds_val = H5ECGDataset(args.path_to_hdf5, df_val, label_cols, dataset_name=args.dataset_name, augment=False)
    ds_tst = H5ECGDataset(args.path_to_hdf5, df_tst, label_cols, dataset_name=args.dataset_name, augment=False)

    # optional: weighted sampler (multi-label → 用逆频率^0.5 生成样本权重)
    sampler = None
    if args.weighted_sampler:
        prev_trn = (Y_trn.mean(axis=0) + 1e-8)
        cls_w = (1.0 / prev_trn) ** 0.5
        sw = 1.0 + (Y_trn * cls_w).max(axis=1)  # 每个样本的最大类权重
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sw), num_samples=len(sw), replacement=True)

    train_loader = DataLoader(
        ds_trn, batch_size=args.batch_size,
        sampler=sampler if sampler is not None else None,
        shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(ds_tst, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # model / loss / opt
    model = ECGMambaNet(n_classes=n_classes, d_model=args.d_model,
                        n_blocks=args.n_blocks, p_drop=args.dropout,
                        bimamba=not args.unidirectional).to(device)

    alpha_t = torch.from_numpy(alpha_vec).to(device)
    criterion = FocalLossLogits(alpha=alpha_t, gamma=args.gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    # sched: reduce on plateau (val macro AUCPR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=2, min_lr=1e-6, verbose=True
    )

    # training
    best_aupr = -1.0
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)   # (B,12,4096)
            yb = yb.to(device, non_blocking=True)   # (B,C)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(xb)                  # (B,C)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)

        trn_loss = running_loss / len(train_loader.dataset)
        # eval
        val_metrics = evaluate(model, val_loader, device, n_classes)
        val_aupr = val_metrics["AUCPR_macro"]
        val_aucroc = val_metrics["AUCROC_macro"]
        scheduler.step(val_aupr)

        log_row = {
            "epoch": epoch,
            "train_loss": trn_loss,
            "val_AUCPR": val_aupr,
            "val_AUCROC": val_aucroc,
            "lr": optimizer.param_groups[0]["lr"],
            "sec": round(time.time() - t0, 2),
        }
        history.append(log_row)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"loss {trn_loss:.4f} | val_AUCPR {val_aupr:.4f} | val_AUCROC {val_aucroc:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | {log_row['sec']}s")

        # save last
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model_last.pth"))

        # early stop on val AUCPR
        improved = (val_aupr is not None) and (val_aupr > best_aupr + 1e-5)
        if improved:
            best_aupr = val_aupr
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pth"))
            with open(os.path.join(args.save_dir, "val_perclass_best.json"), "w") as f:
                json.dump(val_metrics, f, indent=2)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val_AUCPR={best_aupr:.4f}")
                break

        # write csv log
        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, "training_log.csv"), index=False)

    # evaluate on test (load best)
    best_ckpt = os.path.join(args.save_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = evaluate(model, test_loader, device, n_classes)
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Test:", {k: v for k, v in test_metrics.items() if k != "per_class"})


# ------------------------
# Args
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ECG classification with (Bi)Mamba on PTB-XL.")
    parser.add_argument("path_to_hdf5", type=str)
    parser.add_argument("path_to_csv", type=str)
    parser.add_argument("--dataset_name", type=str, default="tracings")
    parser.add_argument("--labels", type=str, default="2dAVB,3dAVB,PAC,PVC,AFLT")

    # model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unidirectional", action="store_true", help="Use single Mamba (default BiMamba).")

    # train
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP).")
    parser.add_argument("--save_dir", type=str, default="./mamba_ckpts")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--weighted_sampler", action="store_true", help="Enable WeightedRandomSampler for train.")

    # focal alpha
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha_mode", type=str, default="train", choices=["train", "val", "const", "file"])
    parser.add_argument("--alpha_const", type=float, default=0.75)
    parser.add_argument("--alpha_file", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
