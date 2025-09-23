# train_head_dualthr_cost.py
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    precision_recall_curve, roc_curve
)
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# 数据集 / 模型
# =========================
class NpzEmbedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]


class MLPHead(nn.Module):
    """与你现有评测脚本一致的小头"""
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
# 训练与基础指标
# =========================
def make_pos_weight(Y: np.ndarray) -> torch.Tensor:
    """
    pos_weight[c] = N_neg / N_pos，避免除零并限幅到[1,50]
    """
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
    """
    返回 macro-AUROC（跳过全正/全负标签） 和 macro-F1(阈值0.5)
    """
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
    """
    返回 macro-AUC, macro-F1, val_BCE；
    若 return_probs=True，同步返回 (Y, P)
    """
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
        # 累计 BCE（对概率的近似）
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
# 标签映射（训练 10 类 → 评测 5 类）
# =========================
DEFAULT_TRAIN_LABELS = ['1AVB','CRBBB','CLBBB','AFIB','STACH','2AVB','3AVB','PAC','PVC','AFLT']
TARGET_CANONICAL = ['1dAVB', 'LBBB', 'RBBB', 'PVC', 'AFLT']
# 将 5 类的规范名映射到训练使用的 10 类标签名
TARGET_TO_TRAINED = {
    "1dAVB": "1AVB",
    "LBBB":  "CLBBB",
    "RBBB":  "CRBBB",
    "PVC":   "PVC",
    "AFLT":  "AFLT",
}


# =========================
# 阈值选择：双阈值 + 代价敏感
# =========================
def pick_dual_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.90,
    target_specificity: float = 0.95,
    thr_min: float = 0.50,
    thr_max: float = 0.99
) -> Tuple[float, float]:
    """
    双阈值（拒判灰区）:
      - τ⁺(阳性阈) 通过 PR 曲线选，使 precision ≥ target_precision
      - τ⁻(阴性阈) 通过 ROC 曲线选，使 specificity ≥ target_specificity
    同时做边界与顺序约束：thr_min ≤ τ⁻ ≤ τ⁺ ≤ thr_max
    """
    # τ⁺：Precision 目标
    precisions, recalls, thr_pr = precision_recall_curve(y_true, y_prob)
    tau_pos = thr_max
    found_pos = False
    # 注意：thr_pr 长度 = len(precisions)-1
    for p, t in zip(precisions[:-1], thr_pr):
        if p >= target_precision:
            tau_pos = float(t)
            found_pos = True
            break
    if not found_pos:
        # 若精确率达不到目标，就选使精确率最高的阈值（退而求其次）
        best_p, best_t = 0.0, thr_max
        for p, t in zip(precisions[:-1], thr_pr):
            if p > best_p:
                best_p, best_t = float(p), float(t)
        tau_pos = float(best_t)

    # τ⁻：Specificity 目标
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    tau_neg = thr_min
    found_neg = False
    for sp, t in zip(specificity, thr_roc):
        if sp >= target_specificity:
            tau_neg = float(t)
            found_neg = True
            break
    if not found_neg:
        # 若特异度达不到目标，就选使特异度最高的阈值
        best_sp, best_t = 0.0, thr_min
        for sp, t in zip(specificity, thr_roc):
            if sp > best_sp:
                best_sp, best_t = float(sp), float(t)
        tau_neg = float(best_t)

    # 约束 & 保守裁剪
    tau_neg = max(thr_min, min(tau_neg, thr_max))
    tau_pos = max(thr_min, min(tau_pos, thr_max))
    if tau_neg > tau_pos:
        # 若出现交叉，则折中
        mid = (tau_neg + tau_pos) / 2.0
        tau_neg, tau_pos = mid, mid

    return float(tau_neg), float(tau_pos)


def pick_threshold_by_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fn: float = 5.0,
    c_fp: float = 1.0,
    thr_min: float = 0.50,
    thr_max: float = 0.99,
    thr_step: float = 0.005
) -> Tuple[float, float]:
    """
    单阈值（代价敏感）：
      最小化 期望代价 = c_fn * FN + c_fp * FP
      搜索区间 [thr_min, thr_max]，步长 thr_step
    返回 (best_thr, best_cost)
    """
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
    target_precision: float = 0.90,
    target_specificity: float = 0.95,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    thr_min: float = 0.50,
    thr_max: float = 0.99,
    thr_step: float = 0.01
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    返回：
      dual_dict: {target: {"tau_neg":..., "tau_pos":...}}
      cost_dict: {target: best_single_threshold_by_cost}
    """
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

        # 双阈值
        tau_neg, tau_pos = pick_dual_thresholds(
            y_true=y, y_prob=p,
            target_precision=target_precision,
            target_specificity=target_specificity,
            thr_min=thr_min, thr_max=thr_max
        )
        dual_dict[tname] = {"tau_neg": tau_neg, "tau_pos": tau_pos}

        # 代价敏感单阈值
        t_cost, best_cost = pick_threshold_by_cost(
            y_true=y, y_prob=p, c_fn=cost_fn, c_fp=cost_fp,
            thr_min=thr_min, thr_max=thr_max, thr_step=thr_step
        )
        cost_dict[tname] = t_cost

        # 打印一下
        print(f"[{tname}] dual thresholds: tau_neg={tau_neg:.3f}, tau_pos={tau_pos:.3f} | cost-thr={t_cost:.3f}")

    return dual_dict, cost_dict


# =========================
# 主流程
# =========================
def main(npz_path: str, out_head: str = "./clf_head.pt",
         epochs=30, bs=256, lr=1e-3, seed=42, val_size=0.15,
         target_precision=0.90, target_specificity=0.95,
         cost_fn=5.0, cost_fp=1.0,
         thr_min=0.50, thr_max=0.99, thr_step=0.01,
         trained_label_order: List[str] = None,
         targets: List[str] = None):
    np.random.seed(seed); torch.manual_seed(seed)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                 # [N,D]
    Y = data["Y"]                 # [N,C]
    labels = list(data["labels"]) if trained_label_order is None else trained_label_order
    print("X:", X.shape, "Y:", Y.shape, "labels:", labels)

    pos_counts = Y.sum(axis=0).astype(int)
    print("Positive count per label:", dict(zip(labels, map(int, pos_counts))))

    # 分层划分
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

    # 用最优权重在验证集上跑一遍，拿到概率，做阈值搜索
    head.load_state_dict(best_state)
    auc, f1, val_bce, Yv, Pv = eval_epoch(head, va_ld, device, return_probs=True)

    # 目标类列表
    if targets is None:
        targets = TARGET_CANONICAL
    print("Target classes for thresholding:", targets)

    # 扫描阈值（双阈值 + 代价敏感）
    dual_dict, cost_dict = scan_thresholds_all_classes_dual_and_cost(
        Y_true=Yv, Y_prob=Pv,
        trained_label_order=labels,
        targets=targets,
        target_precision=target_precision,
        target_specificity=target_specificity,
        cost_fn=cost_fn, cost_fp=cost_fp,
        thr_min=thr_min, thr_max=thr_max, thr_step=thr_step
    )

    # 保存双阈值 JSON
    dual_path = os.path.splitext(out_head)[0] + "_thresholds_dual.json"
    with open(dual_path, "w", encoding="utf-8") as f:
        json.dump(dual_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved per-class dual thresholds to {dual_path}")

    # 保存代价敏感单阈值 JSON
    cost_path = os.path.splitext(out_head)[0] + "_thresholds_cost.json"
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved per-class cost-sensitive thresholds to {cost_path}")

    # 小结打印
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

    # 双阈值目标
    ap.add_argument("--target-precision", type=float, default=0.80,
                    help="Positive decision high-threshold target precision for τ+.")
    ap.add_argument("--target-specificity", type=float, default=0.90,
                    help="Negative decision low-threshold target specificity (1-FPR) for τ-.")

    # 代价敏感
    ap.add_argument("--cost-fn", type=float, default=5.0,
                    help="Cost of FN in cost-sensitive threshold search.")
    ap.add_argument("--cost-fp", type=float, default=1.0,
                    help="Cost of FP in cost-sensitive threshold search.")

    # 阈值扫描区间
    ap.add_argument("--thr-min", type=float, default=0.50)
    ap.add_argument("--thr-max", type=float, default=0.99)
    ap.add_argument("--thr-step", type=float, default=0.01)

    # 可选：自定义训练时的标签顺序（默认读取 NPZ 中的 labels）
    ap.add_argument("--trained-labels", nargs="*", default=None,
                    help="Override label order used during training (10-class order).")
    # 可选：自定义参与阈值搜索的目标类（默认 1dAVB/LBBB/RBBB/PVC/AFLT）
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Target classes to threshold (canonical names).")

    args = ap.parse_args()

    main(
        npz_path=args.npz, out_head=args.out,
        epochs=args.epochs, bs=args.bs, lr=args.lr,
        val_size=args.val_size,
        target_precision=args.target_precision,
        target_specificity=args.target_specificity,
        cost_fn=args.cost_fn, cost_fp=args.cost_fp,
        thr_min=args.thr_min, thr_max=args.thr_max, thr_step=args.thr_step,
        trained_label_order=args.trained_labels,
        targets=args.targets
    )
