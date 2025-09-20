# train_head.py (修复版)
import os
import numpy as np
from typing import List
from sklearn.metrics import f1_score, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class NpzEmbedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]


class MLPHead(nn.Module):
    """比单层更稳一点的小头"""
    def __init__(self, hidden_dim: int, n_classes: int, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p),
            nn.Linear(512, n_classes)
        )
    def forward(self, x): return self.net(x)


def make_pos_weight(Y: np.ndarray) -> torch.Tensor:
    """
    pos_weight[c] = N_neg / N_pos，避免除零。
    """
    N, C = Y.shape
    pos = Y.sum(axis=0)            # [C]
    neg = N - pos
    pw = np.zeros(C, dtype=np.float32)
    for c in range(C):
        if pos[c] < 1:      # 极端情况：全负
            pw[c] = 1.0
        else:
            pw[c] = float(neg[c] / max(pos[c], 1.0))
    # 限幅，避免过大导致训练不稳
    pw = np.clip(pw, 1.0, 50.0)
    return torch.tensor(pw, dtype=torch.float32)


def collate_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    """
    返回可计算标签上的 macro-AUROC 与 macro-F1（阈值0.5）。
    若所有标签都不可算 AUROC，则返回 auc=None。
    """
    C = y_true.shape[1]
    aucs = []
    for c in range(C):
        y = y_true[:, c]
        p = y_prob[:, c]
        if (y.max() == y.min()):  # 单类，跳过
            continue
        try:
            aucs.append(roc_auc_score(y, p))
        except Exception:
            pass
    auc = float(np.mean(aucs)) if len(aucs) > 0 else None

    f1 = f1_score(y_true, (y_prob > 0.5).astype(np.int32), average="macro", zero_division=0)
    return auc, f1


def train_one_epoch(model, loader, crit, optim, device):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    Y = np.concatenate(ys, axis=0)
    P = np.concatenate(ps, axis=0)
    auc, f1 = collate_metrics(Y, P)
    # 同时返回 BCE 损失，便于在 AUC 不可算时用 val_loss 选最优
    bce = float(-(Y * np.log(P + 1e-8) + (1 - Y) * np.log(1 - P + 1e-8)).mean())
    return auc, f1, bce


def main(npz_path: str, out_head: str = "./clf_head.pt",
         epochs=30, bs=256, lr=1e-3, seed=42, val_size=0.15):
    np.random.seed(seed); torch.manual_seed(seed)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                 # [N,D]
    Y = data["Y"]                 # [N,C]
    labels = list(data["labels"])
    print("X:", X.shape, "Y:", Y.shape, "labels:", labels)

    # 打印每个标签的阳性计数，看看是否非常稀有
    pos_counts = Y.sum(axis=0).astype(int)
    print("Positive count per label:", dict(zip(labels, map(int, pos_counts))))

    # 1) 多标签分层划分（关键修复点）
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(msss.split(X, Y))
    Xtr, Xval = X[tr_idx], X[va_idx]
    Ytr, Yval = Y[tr_idx], Y[va_idx]

    tr_ds = NpzEmbedDataset(Xtr, Ytr)
    va_ds = NpzEmbedDataset(Xval, Yval)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = MLPHead(hidden_dim=X.shape[1], n_classes=Y.shape[1]).to(device)

    # 2) 类不平衡加权（关键修复点）
    pos_weight = make_pos_weight(Ytr).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_key = None  # 记录最优的指标（优先 AUC，其次 val_loss 低）
    best_auc = -1.0
    best_loss = float("inf")

    for ep in range(1, epochs + 1):
        train_loss = train_one_epoch(head, tr_ld, crit, optim, device)
        auc, f1, val_bce = eval_epoch(head, va_ld, device)
        auc_str = f"{auc:.4f}" if auc is not None else "n/a"
        print(f"[Epoch {ep:03d}] train_bce={train_loss:.4f}  val_bce={val_bce:.4f}  val_auc={auc_str}  val_f1={f1:.4f}")

        improved = False
        if auc is not None:
            if auc > best_auc:
                improved = True
                best_auc = auc
                best_loss = val_bce
                best_key = f"AUC {auc:.4f}"
        else:
            # 当 AUC 不可算时，用 val_bce 作为指标：更小更好
            if val_bce < best_loss:
                improved = True
                best_loss = val_bce
                best_key = f"VAL_BCE {val_bce:.4f}"

        if improved:
            best_state = head.state_dict().copy()

    if best_state is None:
        # 兜底：保存最后一轮
        best_state = head.state_dict().copy()
        best_key = "last_epoch (fallback)"
    torch.save(best_state, out_head)
    print(f"Saved best head to {out_head}  ({best_key})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to cached embeddings .npz")
    ap.add_argument("--out", default="./clf_head.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-size", type=float, default=0.15)
    args = ap.parse_args()
    main(args.npz, args.out, args.epochs, args.bs, args.lr, val_size=args.val_size)
