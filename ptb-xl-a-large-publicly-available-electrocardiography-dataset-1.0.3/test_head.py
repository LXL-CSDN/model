# test_head.py (per-class thresholds enabled)
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
def hubert_embed(batch_flat: np.ndarray, hubert_model: AutoModel) -> torch.Tensor:
    x = torch.tensor(batch_flat, dtype=torch.float32, device=DEVICE)  # [B,6000]
    out = hubert_model(input_values=x)  # 大多仓库支持这个参数名
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
    Y_true_full = []
    batch_flats = []
    IDS = []

    # 加载 HuBERT 一次
    hubert_model = AutoModel.from_pretrained(args.hubert_id, trust_remote_code=True).to(DEVICE)
    hubert_model.eval()

    # 先占位，稍后根据第一次前向得知 hidden_dim 再加载头
    head = None

    # 真实标签矩阵（按训练顺序的完整 10 类）
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Preprocess"):
        scp = row['scp_codes_dict'] if isinstance(row['scp_codes_dict'], dict) else {}
        y_full = multilabel_row_with_threshold(scp, trained_labels, thr=50.0)
        Y_true_full.append(y_full)

        rec_path = os.path.join(args.records100, row['filename_lr'])
        rec_path = os.path.splitext(rec_path)[0]  # 无扩展名
        try:
            flat, fs = preprocess_wfdb_record(rec_path)
            assert fs == 100 and flat.shape[0] == 6000
            batch_flats.append(flat)
            IDS.append(rec_path)
        except Exception as e:
            print(f"[WARN] skip {rec_path}: {e}")

    if len(batch_flats) == 0:
        raise RuntimeError("没有有效样本被处理，请检查 records 路径与 CSV。")

    # 嵌入（分批以省显存）
    B = args.batch_size
    P_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(batch_flats), B), desc="HuBERT+Head"):
            batch = np.stack(batch_flats[i:i+B], axis=0)  # [b,6000]
            emb = hubert_embed(batch, hubert_model)       # [b,D]

            # 首次才加载分类头
            if head is None:
                hidden_dim = emb.shape[1]
                head = MLPHead(hidden_dim, n_classes=len(trained_labels)).to(DEVICE)
                state = torch.load(args.head, map_location='cpu')
                head.load_state_dict(state)
                head.eval()

            logits = head(emb.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()   # [b,C]
            P_list.append(probs)

    P_full = np.concatenate(P_list, axis=0)
    Y_true_full = np.stack(Y_true_full, axis=0)[:len(IDS)]  # 对齐成功样本数

    # 只抽取 5 类指标
    target_indices = [idx_map[k] for k in TARGET_CANONICAL]
    Y = Y_true_full[:, target_indices]
    P = P_full[:, target_indices]

    # ---- 支持每类阈值（如果提供了 JSON）----
    per_class_thr = None
    if args.thresholds_json and os.path.isfile(args.thresholds_json):
        with open(args.thresholds_json, "r", encoding="utf-8") as f:
            thr_map = json.load(f)  # 需要包含键：1dAVB,LBBB,RBBB,PVC,AFLT
        per_class_thr = np.array([float(thr_map.get(k, args.threshold)) for k in TARGET_CANONICAL], dtype=np.float32)
        print("Use per-class thresholds:", {k: float(v) for k, v in zip(TARGET_CANONICAL, per_class_thr)})
    else:
        print("Use single threshold:", args.threshold)

    # 生成预测
    if per_class_thr is None:
        y_pred = (P >= args.threshold).astype(int)      # [N,5]
    else:
        y_pred = (P >= per_class_thr.reshape(1, -1)).astype(int)

    # 逐类混淆矩阵 (TN,FP,FN,TP)
    cms = []
    for i in range(Y.shape[1]):
        tn, fp, fn, tp = confusion_matrix(Y[:, i], y_pred[:, i], labels=[0,1]).ravel()
        cms.append((tn, fp, fn, tp))

    # 逐类 PR/RC/F1
    prec, rec, f1, support = precision_recall_fscore_support(
        Y, y_pred, average=None, zero_division=0
    )

    # 逐类 ROC-AUC、PR-AUC（基于概率，不受阈值影响）
    aucs = []
    aps  = []
    for i in range(Y.shape[1]):
        aucs.append(safe_auc(Y[:, i], P[:, i]))
        aps.append (safe_ap (Y[:, i], P[:, i]))

    # Macro 平均
    macro_auc = np.mean([a for a in aucs if a is not None]) if any(a is not None for a in aucs) else None
    macro_ap  = np.mean([a for a in aps  if a is not None]) if any(a is not None for a in aps ) else None
    macro_f1  = float(np.mean(f1))
    macro_prec= float(np.mean(prec))
    macro_rec = float(np.mean(rec))

    # 打印结果
    print("\n=== Evaluation on folds {} ===".format(sorted(folds)))
    if per_class_thr is None:
        print("Targets:", TARGET_CANONICAL, "  threshold=", args.threshold)
    else:
        print("Targets:", TARGET_CANONICAL, "  per-class thresholds=", {k: float(v) for k, v in zip(TARGET_CANONICAL, per_class_thr)})

    print("Num samples:", Y.shape[0])

    for i, name in enumerate(TARGET_CANONICAL):
        tn, fp, fn, tp = cms[i]
        auc_s = 'n/a' if aucs[i] is None else f'{aucs[i]:.4f}'
        ap_s  = 'n/a' if aps[i]  is None else f'{aps[i]:.4f}'
        print(f"\n[{name}]")
        print(f"Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"Precision={prec[i]:.4f}  Recall={rec[i]:.4f}  F1={f1[i]:.4f}  Support={int(support[i])}")
        print(f"ROC-AUC={auc_s}  PR-AUC={ap_s}")

    print("\n=== Macro averages over 5 targets ===")
    print(f"Macro Precision={macro_prec:.4f}  Macro Recall={macro_rec:.4f}  Macro F1={macro_f1:.4f}")
    print(f"Macro ROC-AUC={('n/a' if macro_auc is None else f'{macro_auc:.4f}')}"
          f"  Macro PR-AUC={('n/a' if macro_ap  is None else f'{macro_ap:.4f}')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl-csv", required=True, help="Path to ptbxl_database.csv")
    ap.add_argument("--records100", required=True, help="Path to records100 folder (root that contains records100/...)")
    ap.add_argument("--head", required=True, help="Path to trained classifier head .pt")
    ap.add_argument("--hubert-id", default="Edoardo-BS/hubert-ecg-base")
    ap.add_argument("--folds", type=int, nargs="+", default=[9,10], help="Eval folds, e.g. 9 10")
    ap.add_argument("--threshold", type=float, default=0.5, help="Fallback threshold if --thresholds-json not provided")
    ap.add_argument("--thresholds-json", type=str, default=None,
                    help="Path to per-class threshold JSON (keys: 1dAVB,LBBB,RBBB,PVC,AFLT). If set, overrides --threshold.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--labels", nargs="*", default=None, help="Training label order (default uses script's 10-class order)")
    args = ap.parse_args()
    main(args)
