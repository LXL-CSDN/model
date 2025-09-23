# train_head.py (带“每类最优阈值”搜索与保存)
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
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
    N, C = Y.shape
    pos = Y.sum(axis=0)            # [C]
    neg = N - pos
    pw = np.zeros(C, dtype=np.float32)
    for c in range(C):
        if pos[c] < 1:
            pw[c] = 1.0
        else:
            pw[c] = float(neg[c] / max(pos[c], 1.0))
    pw = np.clip(pw, 1.0, 50.0)
    return torch.tensor(pw, dtype=torch.float32)


def collate_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    C = y_true.shape[1]
    aucs = []
    for c in range(C):
        y = y_true[:, c]
        p = y_prob[:, c]
        if (y.max() == y.min()):
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
def eval_epoch(model, loader, device, return_probs: bool = False):
    model.eval()
    ys, ps = [], []
    bce_total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
        # 累计 BCE（用概率近似）
        bce_total += float(-(y.numpy() * np.log(prob + 1e-8) + (1 - y.numpy()) * np.log(1 - prob + 1e-8)).sum())
        n += y.shape[0]
    Y = np.concatenate(ys, axis=0)
    P = np.concatenate(ps, axis=0)
    auc, f1 = collate_metrics(Y, P)
    val_bce = bce_total / n
    if return_probs:
        return auc, f1, val_bce, Y, P
    return auc, f1, val_bce


def scan_best_thresholds(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    trained_label_order: List[str],
    objective: str = "f1",
    thr_min: float = 0.50,
    thr_max: float = 0.95,
    thr_step: float = 0.01
) -> Dict[str, float]:
    """
    在验证集上为指定的 5 个类别（1AVB/LBBB/RBBB/PVC/AFLT）逐一扫描阈值并选最佳。
    - 注意：你训练用的标签名是: ['1AVB','CRBBB','CLBBB','AFIB','STACH','2AVB','3AVB','PAC','PVC','AFLT']
      其中 LBBB=CLBBB, RBBB=CRBBB。
    - objective: "f1" / "precision" / "recall"
    """
    # 训练文件中的名字
    name_to_idx = {name: i for i, name in enumerate(trained_label_order)}

    # 题目要求的 5 类（输出要用的 key）
    target_to_trained = {
        "1dAVB": "1AVB",
        "LBBB":  "CLBBB",
        "RBBB":  "CRBBB",
        "PVC":   "PVC",
        "AFLT":  "AFLT",
    }

    thr_grid = np.arange(thr_min, thr_max + 1e-9, thr_step)
    best_thr: Dict[str, float] = {}

    for out_name, trained_name in target_to_trained.items():
        if trained_name not in name_to_idx:
            print(f"[WARN] {trained_name} not in trained labels, skip.")
            continue
        c = name_to_idx[trained_name]
        y = Y_true[:, c].astype(int)
        p = Y_prob[:, c]

        if y.max() == y.min():
            # 验证集全正或全负，无法通过阈值区分；给一个保守阈值
            best_thr[out_name] = float(thr_min)
            print(f"[{out_name}] val set has a single class; use {thr_min:.2f} by default.")
            continue

        best_score = -1.0
        best_t = thr_min
        for t in thr_grid:
            yhat = (p >= t).astype(int)
            if objective == "f1":
                score = f1_score(y, yhat, zero_division=0)
            elif objective == "precision":
                score = precision_score(y, yhat, zero_division=0)
            elif objective == "recall":
                score = recall_score(y, yhat, zero_division=0)
            else:
                raise ValueError("objective must be one of {'f1','precision','recall'}")
            if score > best_score:
                best_score = score
                best_t = t
        best_thr[out_name] = float(best_t)
        print(f"[{out_name}] best {objective}={best_score:.4f} at threshold={best_t:.2f}")

    return best_thr


def main(npz_path: str, out_head: str = "./clf_head.pt",
         epochs=30, bs=256, lr=1e-3, seed=42, val_size=0.15,
         objective: str = "f1", thr_min=0.50, thr_max=0.95, thr_step=0.01):
    np.random.seed(seed); torch.manual_seed(seed)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                 # [N,D]
    Y = data["Y"]                 # [N,C]
    labels = list(data["labels"])
    print("X:", X.shape, "Y:", Y.shape, "labels:", labels)

    pos_counts = Y.sum(axis=0).astype(int)
    print("Positive count per label:", dict(zip(labels, map(int, pos_counts))))

    # 多标签分层划分
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

    pos_weight = make_pos_weight(Ytr).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_key = None
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
            if val_bce < best_loss:
                improved = True
                best_loss = val_bce
                best_key = f"VAL_BCE {val_bce:.4f}"

        if improved:
            best_state = head.state_dict().copy()

    if best_state is None:
        best_state = head.state_dict().copy()
        best_key = "last_epoch (fallback)"
    torch.save(best_state, out_head)
    print(f"Saved best head to {out_head}  ({best_key})")

    # ===== 用最优权重在验证集上跑一遍，拿到概率，做阈值搜索 =====
    head.load_state_dict(best_state)
    auc, f1, val_bce, Yv, Pv = eval_epoch(head, va_ld, device, return_probs=True)

    best_thr = scan_best_thresholds(
        Y_true=Yv,
        Y_prob=Pv,
        trained_label_order=labels,
        objective=objective,
        thr_min=thr_min,
        thr_max=thr_max,
        thr_step=thr_step
    )

    # 保存阈值（键名按你的评测脚本习惯：1dAVB/LBBB/RBBB/PVC/AFLT）
    thr_path = os.path.splitext(out_head)[0] + "_thresholds.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(best_thr, f, ensure_ascii=False, indent=2)
    print(f"Saved per-class best thresholds to {thr_path}")
    print("Thresholds:", best_thr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to cached embeddings .npz")
    ap.add_argument("--out", default="./clf_head.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--objective", type=str, default="f1", choices=["f1","precision","recall"])
    ap.add_argument("--thr-min", type=float, default=0.50)
    ap.add_argument("--thr-max", type=float, default=0.95)
    ap.add_argument("--thr-step", type=float, default=0.01)
    args = ap.parse_args()

    main(args.npz, args.out, args.epochs, args.bs, args.lr,
         val_size=args.val_size,
         objective=args.objective,
         thr_min=args.thr_min, thr_max=args.thr_max, thr_step=args.thr_step)
