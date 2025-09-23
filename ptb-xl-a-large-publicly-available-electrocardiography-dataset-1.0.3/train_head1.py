import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve
)
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# Dataset / Model
# =========================
class NpzEmbedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]


class MLPHead(nn.Module):
    """Small MLP head consistent with your eval script"""
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


# =========================
# Training utilities
# =========================
DEFAULT_TRAIN_LABELS = ['1AVB','CRBBB','CLBBB','AFIB','STACH','2AVB','3AVB','PAC','PVC','AFLT']
TARGET_CANONICAL = ['1dAVB', 'LBBB', 'RBBB', 'PVC', 'AFLT']
# Map canonical 5-class names to 10-class training labels
TARGET_TO_TRAINED = {
    "1dAVB": "1AVB",
    "LBBB":  "CLBBB",
    "RBBB":  "CRBBB",
    "PVC":   "PVC",
    "AFLT":  "AFLT",
}


def make_pos_weight(Y: np.ndarray) -> torch.Tensor:
    """pos_weight[c] = N_neg / N_pos, clipped to [1, 50]"""
    N, C = Y.shape
    pos = Y.sum(axis=0)
    neg = N - pos
    pw = np.zeros(C, dtype=np.float32)
    for c in range(C):
        if pos[c] < 1:
            pw[c] = 1.0
        else:
            pw[c] = float(neg[c] / max(pos[c], 1.0))
    pw = np.clip(pw, 1.0, 50.0)
    return torch.tensor(pw, dtype=torch.float32)


def collate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Return macro-AUROC (skip degenerate labels) and macro-F1 @ 0.5"""
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
    f1 = f1_score(y_true, (y_prob > 0.5).astype(np.int32),
                  average="macro", zero_division=0)
    return auc, f1


def train_one_epoch(model, loader, crit, optim, device) -> float:
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
    """Return macro-AUC, macro-F1, val_BCE; optionally also (Y, P)."""
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
        bce_total += float(-(y.numpy() * np.log(prob + 1e-8) +
                             (1 - y.numpy()) * np.log(1 - prob + 1e-8)).sum())
        n += y.shape[0]
    Y = np.concatenate(ys, axis=0)
    P = np.concatenate(ps, axis=0)
    auc, f1 = collate_metrics(Y, P)
    val_bce = bce_total / n
    if return_probs:
        return auc, f1, val_bce, Y, P
    return auc, f1, val_bce


# =========================
# Dual thresholds (with NPV) + Cost-sensitive threshold
# =========================

def pick_dual_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision_pos: float = 0.85,   # for tau_pos
    target_npv_neg: float = 0.90,         # for tau_neg via NPV
    thr_min: float = 0.30,
    thr_max: float = 0.99,
    min_gap: float = 0.05
) -> Tuple[float, float]:
    """
    Build a true gray zone: p >= tau_pos => positive; p <= tau_neg => negative; else abstain.

    tau_pos: from PR curve on (y_true, y_prob), choose among precision>=target_precision_pos
             the threshold with the HIGHEST recall (or fallback to best precision in range).

    tau_neg: from PR curve on the NEGATIVE class (y_neg=1-y_true, p_neg=1-y_prob), i.e. NPV curve.
             choose among NPV>=target_npv_neg the threshold with the LARGEST recall_neg (coverage),
             then convert back via tau_neg = 1 - t_neg.

    Finally enforce a minimum gray-zone width by expanding around the midpoint if needed.
    """
    # tau_pos via PR on positives
    P, R, T = precision_recall_curve(y_true, y_prob)  # len(T)=len(P)-1, ascending
    cand_pos = [(t, p, r) for p, r, t in zip(P[:-1], R[:-1], T) if (p >= target_precision_pos and thr_min <= t <= thr_max)]
    if len(cand_pos) == 0:
        # fallback: best precision within bounds
        valid = [(t, p, r) for p, r, t in zip(P[:-1], R[:-1], T) if (thr_min <= t <= thr_max)]
        if len(valid) == 0:
            t_pos = thr_max
        else:
            t_pos, _, _ = max(valid, key=lambda z: z[1])
    else:
        # among feasible set, maximize recall
        t_pos, _, _ = max(cand_pos, key=lambda z: z[2])
    tau_pos = float(np.clip(t_pos, thr_min, thr_max))

    # tau_neg via PR on negatives => NPV curve
    y_neg = 1 - y_true
    p_neg = 1 - y_prob
    Pn, Rn, Tn = precision_recall_curve(y_neg, p_neg)  # Pn is NPV
    cand_neg = [(tn, pn, rn) for pn, rn, tn in zip(Pn[:-1], Rn[:-1], Tn) if (pn >= target_npv_neg)]
    if len(cand_neg) == 0:
        # fallback: highest NPV overall
        tn_best, _, _ = max([(tn, pn, rn) for pn, rn, tn in zip(Pn[:-1], Rn[:-1], Tn)], key=lambda z: z[1], default=(1 - thr_min, 0.0, 0.0))
    else:
        # choose the one with largest recall_neg (coverage of auto-negatives)
        tn_best, _, _ = max(cand_neg, key=lambda z: z[2])
    tau_neg = float(1.0 - tn_best)
    tau_neg = float(np.clip(tau_neg, thr_min, thr_max))

    # enforce a minimal gray-zone width
    if (tau_pos - tau_neg) < min_gap:
        mid = 0.5 * (tau_pos + tau_neg)
        half = 0.5 * min_gap
        tau_neg = max(thr_min, mid - half)
        tau_pos = min(thr_max, mid + half)

    return tau_neg, tau_pos


def pick_threshold_by_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fn: float = 5.0,
    c_fp: float = 1.0,
    thr_min: float = 0.30,
    thr_max: float = 0.99,
    thr_step: float = 0.005
) -> Tuple[float, float]:
    """Single threshold minimizing expected cost = c_fn*FN + c_fp*FP."""
    best_t, best_cost = thr_min, 1e18
    thr_grid = np.arange(thr_min, thr_max + 1e-9, thr_step)
    y = y_true.astype(int)
    for t in thr_grid:
        yhat = (y_prob >= t).astype(int)
        fp = int(np.sum((y == 0) & (yhat == 1)))
        fn = int(np.sum((y == 1) & (yhat == 0)))
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:
            best_cost, best_t = float(cost), float(t)
    return float(best_t), float(best_cost)


def scan_thresholds_all_classes_dual_and_cost(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    trained_label_order: List[str],
    targets: List[str],
    target_precision_pos: float = 0.85,
    target_npv_neg: float = 0.90,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    thr_min: float = 0.30,
    thr_max: float = 0.99,
    thr_step: float = 0.01,
    min_gap: float = 0.05
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    name_to_idx = {name: i for i, name in enumerate(trained_label_order)}
    dual_dict: Dict[str, Dict[str, float]] = {}
    cost_dict: Dict[str, float] = {}

    for tname in targets:
        trained_name = TARGET_TO_TRAINED[tname]
        if trained_name not in name_to_idx:
            print(f"[WARN] {tname}/{trained_name} not found in trained labels; skip.")
            continue
        c = name_to_idx[trained_name]
        y = Y_true[:, c]
        p = Y_prob[:, c]

        # Dual thresholds with NPV + min gray-zone
        tau_neg, tau_pos = pick_dual_thresholds(
            y_true=y, y_prob=p,
            target_precision_pos=target_precision_pos,
            target_npv_neg=target_npv_neg,
            thr_min=thr_min, thr_max=thr_max, min_gap=min_gap
        )
        dual_dict[tname] = {"tau_neg": tau_neg, "tau_pos": tau_pos}

        # Cost-sensitive single threshold
        t_cost, best_cost = pick_threshold_by_cost(
            y_true=y, y_prob=p, c_fn=cost_fn, c_fp=cost_fp,
            thr_min=thr_min, thr_max=thr_max, thr_step=thr_step
        )
        cost_dict[tname] = t_cost

        print(f"[{tname}] dual thresholds: tau_neg={tau_neg:.3f}, tau_pos={tau_pos:.3f} | cost-thr={t_cost:.3f}")

    return dual_dict, cost_dict


# =========================
# Main
# =========================

def main(npz_path: str, out_head: str = "./clf_head.pt",
         epochs=30, bs=256, lr=1e-3, seed=42, val_size=0.15,
         target_precision_pos=0.85, target_npv_neg=0.90,
         cost_fn=5.0, cost_fp=1.0,
         thr_min=0.30, thr_max=0.99, thr_step=0.01, min_gap=0.05,
         trained_label_order: List[str] = None,
         targets: List[str] = None):

    np.random.seed(seed); torch.manual_seed(seed)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    labels = list(data["labels"]) if trained_label_order is None else trained_label_order
    print("X:", X.shape, "Y:", Y.shape, "labels:", labels)

    pos_counts = Y.sum(axis=0).astype(int)
    print("Positive count per label:", dict(zip(labels, map(int, pos_counts))))

    # stratified split
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

    # Run once on val to get probs for threshold search
    head.load_state_dict(best_state)
    auc, f1, val_bce, Yv, Pv = eval_epoch(head, va_ld, device, return_probs=True)

    # targets
    if targets is None:
        targets = TARGET_CANONICAL
    print("Target classes for thresholding:", targets)

    # scan thresholds (dual + cost)
    dual_dict, cost_dict = scan_thresholds_all_classes_dual_and_cost(
        Y_true=Yv, Y_prob=Pv,
        trained_label_order=labels,
        targets=targets,
        target_precision_pos=target_precision_pos,
        target_npv_neg=target_npv_neg,
        cost_fn=cost_fn, cost_fp=cost_fp,
        thr_min=thr_min, thr_max=thr_max, thr_step=thr_step, min_gap=min_gap
    )

    # Save JSONs
    dual_path = os.path.splitext(out_head)[0] + "_thresholds_dual.json"
    with open(dual_path, "w", encoding="utf-8") as f:
        json.dump(dual_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved per-class dual thresholds to {dual_path}")

    cost_path = os.path.splitext(out_head)[0] + "_thresholds_cost.json"
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved per-class cost-sensitive thresholds to {cost_path}")

    print("\n=== Summary ===")
    for k in targets:
        dt = dual_dict.get(k, {})
        ct = cost_dict.get(k, None)
        print(f"{k:6s} | dual: {dt} | cost-thr: {ct}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to cached embeddings .npz")
    ap.add_argument("--out", default="./clf_head.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-size", type=float, default=0.15)

    # Dual thresholds targets
    ap.add_argument("--target-precision-pos", type=float, default=0.85,
                    help="Target precision for tau_pos (positive decision).")
    ap.add_argument("--target-npv-neg", type=float, default=0.90,
                    help="Target NPV for tau_neg (negative decision via NPV).")

    # Cost-sensitive
    ap.add_argument("--cost-fn", type=float, default=5.0,
                    help="Cost of FN in cost-sensitive threshold search.")
    ap.add_argument("--cost-fp", type=float, default=1.0,
                    help="Cost of FP in cost-sensitive threshold search.")

    # Threshold scan range
    ap.add_argument("--thr-min", type=float, default=0.30)
    ap.add_argument("--thr-max", type=float, default=0.99)
    ap.add_argument("--thr-step", type=float, default=0.01)
    ap.add_argument("--min-gap", type=float, default=0.05,
                    help="Minimum gray-zone width (tau_pos - tau_neg) to enforce.")

    # Optional overrides
    ap.add_argument("--trained-labels", nargs="*", default=None,
                    help="Override label order used during training (10-class order).")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Target canonical classes to threshold.")

    args = ap.parse_args()

    main(
        npz_path=args.npz, out_head=args.out,
        epochs=args.epochs, bs=args.bs, lr=args.lr,
        val_size=args.val_size,
        target_precision_pos=args.target_precision_pos,
        target_npv_neg=args.target_npv_neg,
        cost_fn=args.cost_fn, cost_fp=args.cost_fp,
        thr_min=args.thr_min, thr_max=args.thr_max, thr_step=args.thr_step,
        min_gap=args.min_gap,
        trained_label_order=args.trained_labels,
        targets=args.targets
    )