#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from scipy import signal
from transformers import AutoModel

# -------------------- 固定的 10 类顺序（与你训练头一致） --------------------
TARGET_LABELS = ["1AVB","CRBBB","CLBBB","AFIB","STACH","2AVB","3AVB","PAC","PVC","AFLT"]

# AHA 主语句码 -> 10 类标签映射（根据你前面规范整理）
AHA2TARGETS: Dict[str, str] = {
    # AVB
    "82":  "1AVB",   # Prolonged PR
    "83":  "2AVB",   # Mobitz I
    "84":  "2AVB",   # Mobitz II
    "85":  "2AVB",   # 2:1 block
    "88":  "3AVB",   # Complete AV block
    # Bundle branch block (complete)
    "106": "CRBBB",
    "104": "CLBBB",
    # Atrial rhythms
    "50":  "AFIB",
    "51":  "AFLT",
    "30":  "PAC",
    "31":  "PAC",
    # Rate
    "21":  "STACH",
    # PVC
    "60":  "PVC",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 你的 MLPHeadStrong 定义（与训练一致） --------------------
class MLPHeadStrong(nn.Module):
    def __init__(self, hidden_dim: int, n_classes: int, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p),
            nn.Linear(512, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# -------------------- 信号预处理（与此前管线对齐） --------------------
def bandpass_fir(sig: np.ndarray, fs: int, low=0.05, high=47.0) -> np.ndarray:
    x = np.atleast_2d(sig).astype(np.float32)
    T = x.shape[0]
    nyq = fs / 2
    high = min(high, nyq * 0.98)

    target_taps = int(3 * fs)
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
    m = float(np.max(np.abs(sig))) if sig.size else 0.0
    if m > 0:
        sig = sig / m
    return np.clip(sig, -1.0, 1.0).astype(np.float32)

def crop_center_5s(sig: np.ndarray, fs: int) -> np.ndarray:
    target = 5 * fs  # 500 点
    T = sig.shape[0]
    if T == target: return sig
    if T > target:
        s = (T - target) // 2
        return sig[s:s+target]
    pad_front = (target - T) // 2
    pad_back  = target - T - pad_front
    return np.pad(sig, ((pad_front, pad_back),(0,0)), mode="constant")

# -------------------- 读取 SPH h5 --------------------
def read_sph_h5(h5_path: str) -> Tuple[np.ndarray, int]:
    with h5py.File(h5_path, "r") as f:
        dset = None
        for k in ["ecg","ECG","signal","data"]:
            if k in f:
                dset = f[k][:]
                break
        if dset is None:
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    dset = f[k][:]
                    break
        if dset is None:
            raise ValueError("No ECG dataset found.")

        ecg = np.array(dset)
        if ecg.ndim != 2:
            raise ValueError(f"Unexpected ECG ndim={ecg.ndim}")
        if ecg.shape[0] == 12 and ecg.shape[1] != 12:
            ecg = ecg.T   # (12,L)->(L,12)
        elif ecg.shape[1] == 12:
            pass
        else:
            raise ValueError(f"Bad ECG shape: {ecg.shape}")

        fs = 500
        for ak in ["sampling_rate","fs","sample_rate","SamplingRate"]:
            if ak in f.attrs:
                try:
                    fs = int(f.attrs[ak]); break
                except Exception:
                    pass
    return ecg.astype(np.float32), fs

# -------------------- AHA_Code -> 10 类多标签 --------------------
def aha_to_targets(aha_code: str) -> np.ndarray:
    y = np.zeros(len(TARGET_LABELS), dtype=np.float32)
    if aha_code is None: return y
    s = str(aha_code).strip()
    if not s: return y
    idx = {name:i for i,name in enumerate(TARGET_LABELS)}
    for stmt in s.split(";"):
        stmt = stmt.strip()
        if not stmt: continue
        primary = stmt.split("+")[0].strip()
        lab = AHA2TARGETS.get(primary)
        if lab is not None:
            y[idx[lab]] = 1.0
    return y

# -------------------- 评估指标 --------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold=0.5):
    y_bin = (y_prob >= threshold).astype(np.float32)
    m = {}
    m["subset_accuracy"] = float(accuracy_score(y_true, y_bin))
    m["hamming_score"]  = float((y_true == y_bin).mean())
    for avg in ["samples","macro","micro","weighted"]:
        m[f"precision_{avg}"] = float(precision_score(y_true, y_bin, average=avg, zero_division=0))
        m[f"recall_{avg}"]    = float(recall_score(y_true, y_bin, average=avg, zero_division=0))
        m[f"f1_{avg}"]        = float(f1_score(y_true, y_bin, average=avg, zero_division=0))
    try:
        valid = y_true.sum(axis=0) > 0
        if valid.any():
            m["auc_macro"]    = float(roc_auc_score(y_true[:, valid], y_prob[:, valid], average="macro"))
            m["auc_weighted"] = float(roc_auc_score(y_true[:, valid], y_prob[:, valid], average="weighted"))
        else:
            m["auc_macro"] = None; m["auc_weighted"] = None
    except Exception as e:
        print(f"[Warn] AUC failed: {e}")
        m["auc_macro"] = None; m["auc_weighted"] = None
    return m, y_bin

def per_class_report(y_true: np.ndarray, y_bin: np.ndarray):
    rows = []
    for i, name in enumerate(TARGET_LABELS):
        yt = y_true[:, i]
        yp = y_bin[:, i]
        support = int(yt.sum())
        if support == 0: continue
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        rows.append({
            "Label": name, "Support": support,
            "Precision": float(precision_score(yt, yp, zero_division=0)),
            "Recall": float(recall_score(yt, yp, zero_division=0)),
            "F1-Score": float(f1_score(yt, yp, zero_division=0)),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
        })
    return pd.DataFrame(rows)

# -------------------- 主流程（在线嵌入 + 评估） --------------------
@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 读取 metadata
    meta = pd.read_csv(args.metadata)
    id_col = args.id_col
    aha_col = args.aha_col
    if id_col not in meta.columns or aha_col not in meta.columns:
        raise ValueError(f"metadata must contain columns: {id_col}, {aha_col}")
    meta = meta.reset_index(drop=True)
    N = len(meta)
    start = max(0, args.start)
    end = N if args.end is None else min(args.end, N)
    meta = meta.iloc[start:end].copy()

    # HuBERT-ECG
    hubert = AutoModel.from_pretrained(args.hubert_id, trust_remote_code=True).to(DEVICE)
    hubert.eval()

    # 重建 MLPHeadStrong（hidden_dim 待首个 batch 推断）
    head = None
    state = torch.load(args.pt, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("Expect state_dict in clf_head_6.pt")

    y_true_all, y_prob_all = [], []
    kept = 0

    # 批处理
    for i in tqdm(range(0, len(meta), args.bs), desc="Evaluating"):
        batch = meta.iloc[i:i+args.bs]
        flats, Ys = [], []

        # --- 读取 & 预处理 & 展平 + 标签 ---
        for _, row in batch.iterrows():
            ecg_id = str(row[id_col])
            h5p = os.path.join(args.h5_dir, f"{ecg_id}.h5")
            if not os.path.exists(h5p):
                continue
            try:
                sig, fs0 = read_sph_h5(h5p)                # [T,12], fs0
                sig = bandpass_fir(sig, fs0, 0.05, 47.0)
                sig, fs = resample_to_100hz(sig, fs0)
                sig = scale_to_unit(sig)
                sig = crop_center_5s(sig, fs)              # [500,12]
                flat = sig.reshape(-1).astype(np.float32)  # 6000
                if flat.shape[0] != 6000:
                    continue
                flats.append(flat)
                Ys.append(aha_to_targets(row[aha_col]))
            except Exception:
                continue

        if not flats:
            continue

        flats = np.stack(flats, axis=0)   # [B,6000]
        X = torch.tensor(flats, dtype=torch.float32, device=DEVICE)

        # --- HuBERT 前向（mean pool） ---
        out = hubert(input_values=X)
        hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        emb = hs.mean(dim=1)              # [B, D]
        D = emb.shape[1]

        # --- 构建并加载分类头（一次性） ---
        if head is None:
            head = MLPHeadStrong(hidden_dim=D, n_classes=len(TARGET_LABELS), p=args.dropout).to(DEVICE)
            head.load_state_dict(state, strict=True)
            head.eval()

        # --- 预测 ---
        logits = head(emb)                # [B, 10]
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_true_all.append(np.stack(Ys, axis=0))
        y_prob_all.append(prob)
        kept += prob.shape[0]

        # 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if kept == 0:
        raise RuntimeError("No valid samples processed. Check paths & metadata.")

    Y_true = np.vstack(y_true_all)
    Y_prob = np.vstack(y_prob_all)

    # 阈值（可选）
    thr_dict = None
    if args.thr_json and os.path.exists(args.thr_json):
        with open(args.thr_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        # 支持 {'1AVB':0.6, ...} 或 {'0':0.5, ...}
        thr_dict = np.full((len(TARGET_LABELS),), 0.5, dtype=np.float32)
        name2idx = {n:i for i,n in enumerate(TARGET_LABELS)}
        for k,v in j.items():
            if k in name2idx:
                thr_dict[name2idx[k]] = float(v)
            elif k.isdigit() and int(k) < len(TARGET_LABELS):
                thr_dict[int(k)] = float(v)

    if thr_dict is None:
        metrics, y_bin = compute_metrics(Y_true, Y_prob, threshold=0.5)
    else:
        y_bin = (Y_prob >= thr_dict.reshape(1,-1)).astype(np.float32)
        metrics, _ = compute_metrics(Y_true, Y_prob, threshold=float(thr_dict.mean()))

    per_cls = per_class_report(Y_true, y_bin)

    # 保存
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "overall_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    per_cls.to_csv(os.path.join(args.out_dir, "per_class_metrics.csv"), index=False)
    np.save(os.path.join(args.out_dir, "raw_probs.npy"), Y_prob)
    np.save(os.path.join(args.out_dir, "y_true.npy"), Y_true)

    # 打印摘要
    print("\n===== Overall =====")
    for k in ["subset_accuracy","hamming_score","f1_samples","f1_macro","f1_micro","f1_weighted","auc_macro","auc_weighted"]:
        v = metrics.get(k); print(f"{k:>16}: {v if v is None else f'{v:.4f}'}")
    if not per_cls.empty:
        print("\nTop by support:\n", per_cls.sort_values("Support", ascending=False).head(10).to_string(index=False))
    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate clf_head_6.pt directly from SPH h5 using HuBERT-ECG embeddings (on-the-fly)")
    ap.add_argument("--metadata", required=True, help="Path to metadata.csv (must include ECG_ID and AHA_Code)")
    ap.add_argument("--h5_dir",   required=True, help="Directory containing <ECG_ID>.h5 files")
    ap.add_argument("--pt",       required=True, help="Path to clf_head_6.pt (state_dict of MLPHeadStrong)")
    ap.add_argument("--out_dir",  default="./eval_head6_from_h5")
    ap.add_argument("--hubert_id", default="Edoardo-BS/hubert-ecg-base", help="HuBERT-ECG model id")
    ap.add_argument("--aha_col",  default="AHA_Code")
    ap.add_argument("--id_col",   default="ECG_ID")
    ap.add_argument("--bs",       type=int, default=64, help="Batch size for embedding+inference")
    ap.add_argument("--dropout",  type=float, default=0.2, help="Dropout p used to rebuild head (match training)")
    ap.add_argument("--thr_json", type=str, default=None, help="Optional thresholds JSON produced at training")
    ap.add_argument("--start",    type=int, default=0)
    ap.add_argument("--end",      type=int, default=None)
    args = ap.parse_args()
    main(args)
