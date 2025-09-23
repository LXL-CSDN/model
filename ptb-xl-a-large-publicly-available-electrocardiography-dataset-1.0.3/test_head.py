import os, json, ast, argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel
import wfdb
from scipy import signal
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

# ------------------------
# 基本配置与工具
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 训练时的标签顺序（如与你不同，请用 --labels 参数覆盖）
DEFAULT_LABELS = ['1AVB','CRBBB','CLBBB','AFIB','STACH','2AVB','3AVB','PAC','PVC','AFLT']

# 目标评测的 5 类（允许同义名映射）
TARGET_CANONICAL = ['1dAVB', 'LBBB', 'RBBB', 'PVC', 'AFLT']
SYN_MAP = {
    '1dAVB': ['1dAVB', '1AVB'],
    'RBBB':  ['RBBB', 'CRBBB'],
    'LBBB':  ['LBBB', 'CLBBB'],
    'PVC':   ['PVC'],
    'AFLT':  ['AFLT']
}

def parse_scp_dict(s: str) -> Dict[str, float]:
    """PTB-XL 的 scp_codes 是 dict 字符串；解析为 dict[str,float]"""
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return {}

def multilabel_row_with_threshold(scp: Dict[str, float], label_names: List[str], thr: float = 50.0) -> np.ndarray:
    """当 scp_codes[key] > thr 才认为该标签为 1，否则为 0"""
    y = np.zeros(len(label_names), dtype=np.float32)
    for i, name in enumerate(label_names):
        if name in scp and float(scp[name]) > thr:
            y[i] = 1.0
    return y

# ------------------------
# 论文一致的预处理：带通→100Hz→[-1,1]→5秒→扁平化
# ------------------------
def bandpass_fir(sig: np.ndarray, fs: int, low=0.05, high=47.0) -> np.ndarray:
    x = np.atleast_2d(sig).astype(np.float32)  # [T,C]
    T = x.shape[0]
    nyq = fs / 2.0
    high = min(high, nyq * 0.98)

    target_taps = int(3 * fs)           # ~3秒
    max_taps = max(5, (T // 3) - 1)
    numtaps = min(target_taps, max_taps)
    if numtaps % 2 == 0: numtaps -= 1
    numtaps = max(numtaps, 5)

    b = signal.firwin(numtaps, [low/nyq, high/nyq], pass_zero=False)

    y = np.empty_like(x)
    default_padlen = 3 * (len(b) - 1)
    padlen = min(default_padlen, T - 1) if T > 1 else 0

    for c in range(x.shape[1]):
        if T <= 3:
            y[:, c] = x[:, c]
        else:
            y[:, c] = signal.filtfilt(b, [1.0], x[:, c], padlen=padlen)
    return y

def resample_to_100hz(sig: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
    if fs == 100:
        return sig.astype(np.float32), 100
    new_len = int(round(sig.shape[0] * 100 / fs))
    out = np.zeros((new_len, sig.shape[1]), dtype=np.float32)
    for c in range(sig.shape[1]):
        out[:, c] = signal.resample(sig[:, c], new_len)
    return out, 100

def scale_to_unit(sig: np.ndarray) -> np.ndarray:
    sig = sig - np.median(sig, axis=0, keepdims=True)
    m = np.max(np.abs(sig))
    if m > 0: sig = sig / m
    return np.clip(sig, -1.0, 1.0).astype(np.float32)

def crop_center_5s(sig: np.ndarray, fs: int) -> np.ndarray:
    target = 5 * fs
    T = sig.shape[0]
    if T == target: return sig
    if T > target:
        s = (T - target) // 2
        return sig[s:s+target]
    pad_front = (target - T) // 2
    pad_back  = target - T - pad_front
    return np.pad(sig, ((pad_front, pad_back),(0,0)), mode="constant")

def preprocess_wfdb_record(rec_path_no_ext: str) -> Tuple[np.ndarray, int]:
    sig, meta = wfdb.rdsamp(rec_path_no_ext)  # sig: [T, 12]
    fs = int(meta["fs"])
    bp = bandpass_fir(sig, fs, 0.05, 47.0)
    sig100, fs100 = resample_to_100hz(bp, fs)
    unit = scale_to_unit(sig100)
    s5 = crop_center_5s(unit, fs100)  # [500, 12]
    flat = s5.reshape(-1).astype(np.float32)   # 500*12=6000
    return flat, 100

# ------------------------
# 分类头（与训练时一致）
# ------------------------
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

# ------------------------
# HuBERT 推理：输入 [B,6000]，输出 [B,D] embedding
# ------------------------
@torch.no_grad()
def hubert_embed(batch_flat: np.ndarray, hubert_id: str) -> torch.Tensor:
    model = AutoModel.from_pretrained(hubert_id, trust_remote_code=True).to(DEVICE)
    model.eval()
    x = torch.tensor(batch_flat, dtype=torch.float32, device=DEVICE)  # [B,6000]
    out = model(input_values=x)  # 大多仓库支持这个参数名
    hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    emb = hs.mean(dim=1)  # [B,D]
    return emb

# ------------------------
# 评测与打印
# ------------------------
def map_target_to_trained_indices(trained_labels: List[str]) -> Dict[str, int]:
    idx_map = {}
    for canon in TARGET_CANONICAL:
        synonyms = SYN_MAP[canon]
        found = None
        for s in synonyms:
            if s in trained_labels:
                found = trained_labels.index(s)
                break
        if found is None:
            raise ValueError(f"找不到 {canon} 在训练标签中的同义名 {synonyms}，请检查 --labels 是否与训练一致。")
        idx_map[canon] = found
    return idx_map

def per_class_confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5):
    """
    y_true,y_prob: [N,C]；返回每类的 (TN, FP, FN, TP)
    """
    y_pred = (y_prob >= thr).astype(int)
    N, C = y_true.shape
    cms = []
    for c in range(C):
        tn, fp, fn, tp = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0,1]).ravel()
        cms.append((tn, fp, fn, tp))
    return cms

def safe_auc(y, p):
    try:
        if len(np.unique(y)) < 2: return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None

def safe_ap(y, p):
    try:
        if len(np.unique(y)) < 2: return None
        return float(average_precision_score(y, p))
    except Exception:
        return None

def main(args):
    trained_labels = args.labels if args.labels else DEFAULT_LABELS
    print("Trained labels order:", trained_labels)
    idx_map = map_target_to_trained_indices(trained_labels)
    print("Target→Index:", idx_map)

    # 读取 PTB-XL 元数据
    df = pd.read_csv(args.ptbxl_csv)
    df = df[df['filename_lr'].notna()].copy()
    df['scp_codes_dict'] = df['scp_codes'].apply(parse_scp_dict)

    # 选第 9/10 折
    folds = set(args.folds)
    df_eval = df[df['strat_fold'].isin(folds)].reset_index(drop=True)
    print(f"Eval samples (folds {sorted(folds)}):", len(df_eval))

    # 构造路径与标签（按“分数>50 计为阳性”）
    Y_all = []
    P_all = []
    IDS = []

    # 先把所有样本的扁平化信号组成批次送入 HuBERT（分批减少显存）
    batch_flats = []
    id_batch = []

    # 真实标签矩阵（按训练顺序的完整 10 类）
    Y_true_full = []

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Preprocess"):
        scp = row['scp_codes_dict'] if isinstance(row['scp_codes_dict'], dict) else {}
        y_full = multilabel_row_with_threshold(scp, trained_labels, thr=50.0)
        # 最终只评测 5 类，但先存全量（后面再索引）
        Y_true_full.append(y_full)

        rec_path = os.path.join(args.records100, row['filename_lr'])
        rec_path = os.path.splitext(rec_path)[0]  # 无扩展名
        try:
            flat, fs = preprocess_wfdb_record(rec_path)
            assert fs == 100 and flat.shape[0] == 6000
            batch_flats.append(flat)
            id_batch.append(rec_path)
            # 分批推理
            if len(batch_flats) == args.batch_size:
                emb = hubert_embed(np.stack(batch_flats, 0), args.hubert_id)  # [B,D]
                if 'head_hidden' in args.__dict__ and args.head_hidden is not None:
                    hidden_dim = args.head_hidden
                else:
                    hidden_dim = emb.shape[1]
                # 加载一次头并缓存
                if 'head' not in globals():
                    global head
                    head = MLPHead(hidden_dim, n_classes=len(trained_labels)).to(DEVICE)
                    state = torch.load(args.head, map_location='cpu')
                    head.load_state_dict(state); head.eval()

                with torch.no_grad():
                    logits = head(emb.to(DEVICE))
                    probs = torch.sigmoid(logits).cpu().numpy()  # [B,C]

                P_all.append(probs)
                IDS.extend(id_batch)
                batch_flats, id_batch = [], []
        except Exception as e:
            print(f"[WARN] skip {rec_path}: {e}")

    # 处理最后一小批
    if batch_flats:
        emb = hubert_embed(np.stack(batch_flats, 0), args.hubert_id)
        hidden_dim = emb.shape[1]
        if 'head' not in locals():
            head = MLPHead(hidden_dim, n_classes=len(trained_labels)).to(DEVICE)
            state = torch.load(args.head, map_location='cpu')
            head.load_state_dict(state); head.eval()
        with torch.no_grad():
            logits = head(emb.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
        P_all.append(probs)
        IDS.extend(id_batch)

    if len(P_all) == 0:
        raise RuntimeError("没有有效样本被处理，请检查 records 路径与 CSV。")

    Y_true_full = np.stack(Y_true_full, axis=0)[:len(IDS)]     # 对齐成功样本数
    P_full = np.concatenate(P_all, axis=0)

    # 只抽取 5 类指标
    target_indices = [idx_map[k] for k in TARGET_CANONICAL]
    Y = Y_true_full[:, target_indices]
    P = P_full[:, target_indices]

    # 逐类混淆矩阵 (TN,FP,FN,TP)
    cms = per_class_confusion(Y, P, thr=args.threshold)

    # 逐类 PR/RC/F1、ROC-AUC、PR-AUC
    prec, rec, f1, support = precision_recall_fscore_support(
        Y, (P >= args.threshold).astype(int),
        average=None, zero_division=0
    )

    aucs = [safe_auc(Y[:, i], P[:, i]) for i in range(Y.shape[1])]
    aps  = [safe_ap (Y[:, i], P[:, i]) for i in range(Y.shape[1])]

    # Macro 平均（忽略 None）
    macro_auc = np.mean([a for a in aucs if a is not None]) if any(a is not None for a in aucs) else None
    macro_ap  = np.mean([a for a in aps  if a is not None]) if any(a is not None for a in aps ) else None
    macro_f1  = float(np.mean(f1))
    macro_prec= float(np.mean(prec))
    macro_rec = float(np.mean(rec))

    # 打印结果
    print("\n=== Evaluation on folds {} ===".format(sorted(folds)))
    print("Targets:", TARGET_CANONICAL, "  threshold=", args.threshold)
    print("Num samples:", Y.shape[0])

    for i, name in enumerate(TARGET_CANONICAL):
        tn, fp, fn, tp = cms[i]
        print(f"\n[{name}]")
        print(f"Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"Precision={prec[i]:.4f}  Recall={rec[i]:.4f}  F1={f1[i]:.4f}  Support={int(support[i])}")
        print(f"ROC-AUC={('n/a' if aucs[i] is None else f'{aucs[i]:.4f}')}"
              f"  PR-AUC={('n/a' if aps[i]  is None else f'{aps[i]:.4f}')}")

    print("\n=== Macro averages over 5 targets ===")
    print(f"Macro Precision={macro_prec:.4f}  Macro Recall={macro_rec:.4f}  Macro F1={macro_f1:.4f}")
    print(f"Macro ROC-AUC={('n/a' if macro_auc is None else f'{macro_auc:.4f}')}"
          f"  Macro PR-AUC={('n/a' if macro_ap  is None else f'{macro_ap:.4f}')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl-csv", required=True, help="Path to ptbxl_database.csv")
    ap.add_argument("--records100", required=True, help="Path to records100 folder")
    ap.add_argument("--head", required=True, help="Path to trained classifier head .pt")
    ap.add_argument("--hubert-id", default="Edoardo-BS/hubert-ecg-base")
    ap.add_argument("--folds", type=int, nargs="+", default=[9,10], help="Eval folds, e.g. 9 10")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--labels", nargs="*", default=None, help="Training label order (default uses script's 10-class order)")
    args = ap.parse_args()
    main(args)
