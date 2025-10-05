#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

import torch
from transformers import AutoModel
from scipy import signal

# ========== 配置 ==========
TARGET_LABELS = ["1AVB","CRBBB","CLBBB","AFIB","STACH","2AVB","3AVB","PAC","PVC","AFLT"]

# AHA 主语句码 -> 10 类标签映射（任何一个命中即置 1）
AHA2TARGETS: Dict[str, str] = {
    # 1AVB
    "82":  "1AVB",   # Prolonged PR interval
    # RBBB / LBBB
    "106": "CRBBB",  # Right bundle-branch block（complete）
    "104": "CLBBB",  # Left bundle-branch block（complete）
    # 心房性：AF、AFLT、PAC
    "50":  "AFIB",   # Atrial fibrillation
    "51":  "AFLT",   # Atrial flutter
    "30":  "PAC",    # Atrial premature complex(es)
    "31":  "PAC",    # Atrial premature complexes, nonconducted
    # PVC
    "60":  "PVC",    # Ventricular premature complex(es)
    # AVB 二度/三度
    "83":  "2AVB",   # Second-degree AV block, Mobitz I (Wenckebach)
    "84":  "2AVB",   # Second-degree AV block, Mobitz II
    "85":  "2AVB",   # 2:1 AV block
    "88":  "3AVB",   # Complete (third-degree) AV block
    # 其它（如需要可扩展）
    # "21": "STACH", # Sinus tachycardia（这类你通常在 PTB-XL 也训练了，若 SPH 要包含，可放开）
    # 为了对齐 TARGET_LABELS 我们也把 STACH 映射上
    "21":  "STACH",  # Sinus tachycardia
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUBERT_ID = "Edoardo-BS/hubert-ecg-base"  # 也可改为 large
TRUST_REMOTE_CODE = True

# ========== 信号预处理（与您 PTB-XL 管线一致） ==========
def bandpass_fir(sig: np.ndarray, fs: int, low=0.05, high=47.0) -> np.ndarray:
    """FIR 带通 + filtfilt；sig: [T, C]"""
    x = np.atleast_2d(sig).astype(np.float32)
    T = x.shape[0]
    nyq = fs / 2
    high = min(high, nyq * 0.98)

    target_taps = int(3 * fs)         # ~3 秒窗
    max_taps = max(5, (T // 3) - 1)   # 避免 padlen 问题
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
    target = 5 * fs
    T = sig.shape[0]
    if T == target: return sig
    if T > target:
        s = (T - target) // 2
        return sig[s:s+target]
    pad_front = (target - T) // 2
    pad_back  = target - T - pad_front
    return np.pad(sig, ((pad_front, pad_back),(0,0)), mode="constant")

# ========== 读取 SPH 的 .h5 ==========
def read_sph_h5(h5_path: str) -> Tuple[np.ndarray, int]:
    """
    返回 (sig[T,12], fs)
    - 支持 ecg 形状 (12,L) 或 (L,12)
    - fs 从 attrs['sampling_rate']（或 'fs','sample_rate'）读取，默认 500
    """
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
            ecg = ecg.T  # (12,L)->(L,12)
        elif ecg.shape[1] == 12:
            pass
        else:
            raise ValueError(f"Bad ECG shape: {ecg.shape}")

        fs = 500
        for ak in ["sampling_rate","fs","sample_rate","SamplingRate"]:
            if ak in f.attrs:
                try:
                    fs = int(f.attrs[ak])
                    break
                except Exception:
                    pass
    return ecg.astype(np.float32), fs

# ========== AHA_Code → 10 类多标签 ==========
def parse_aha_to_targets(aha_code: str) -> np.ndarray:
    """
    AHA_Code 形如 '21+308;104;50'。仅取每条语句 '+' 前的主语句码，并按 AHA2TARGETS 映射。
    """
    y = np.zeros(len(TARGET_LABELS), dtype=np.float32)
    if aha_code is None:
        return y
    s = str(aha_code).strip()
    if not s:
        return y
    tgt_idx = {name:i for i,name in enumerate(TARGET_LABELS)}
    for stmt in s.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        primary = stmt.split("+")[0].strip()
        if primary in AHA2TARGETS:
            lab = AHA2TARGETS[primary]
            y[tgt_idx[lab]] = 1.0
    return y

# ========== HuBERT-ECG 前向（mean pool） ==========
@torch.no_grad()
def hubert_embed(batch_flat: np.ndarray, model: AutoModel) -> np.ndarray:
    """
    batch_flat: [B, 6000] in [-1,1]
    return: [B, D] mean pooled embeddings
    """
    x = torch.tensor(batch_flat, dtype=torch.float32, device=DEVICE)
    out = model(input_values=x)
    hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    emb = hs.mean(dim=1)  # [B, D]
    return emb.detach().cpu().numpy()

# ========== 主流程 ==========
def main(
    metadata_csv: str,
    h5_dir: str,
    out_npz: str,
    aha_col: str = "AHA_Code",
    ecg_id_col: str = "ECG_ID",
    batch_size: int = 64,
    min_keep: int = 1,
    start: int = 0,
    end: Optional[int] = None
):
    """
    - metadata_csv: 含 ECG_ID, AHA_Code
    - h5_dir: 放置 A00001.h5 ... 的目录
    - 批量范围 [start, end) 可用于分块跑
    """
    df = pd.read_csv(metadata_csv)
    if ecg_id_col not in df.columns or aha_col not in df.columns:
        raise ValueError(f"metadata must contain columns: {ecg_id_col}, {aha_col}")

    # 对齐范围
    df = df.reset_index(drop=True)
    if end is None or end > len(df):
        end = len(df)
    df = df.iloc[start:end].copy()

    # 预建标签与路径
    paths, labels = [], []
    ids = []
    for _, row in df.iterrows():
        ecg_id = str(row[ecg_id_col])
        aha = row[aha_col]
        y = parse_aha_to_targets(aha)
        h5_path = os.path.join(h5_dir, f"{ecg_id}.h5")
        paths.append(h5_path)
        labels.append(y)
        ids.append(ecg_id)
    Y_all = np.stack(labels, axis=0)  # N x C

    # 加载 HuBERT-ECG
    model = AutoModel.from_pretrained(HUBERT_ID, trust_remote_code=TRUST_REMOTE_CODE).to(DEVICE)
    model.eval()

    embs, kept_Y, kept_ids = [], [], []
    N = len(paths)
    for i in tqdm(range(0, N, batch_size), desc="Embedding SPH"):
        batch_paths = paths[i:i+batch_size]
        batch_Y = Y_all[i:i+batch_size]
        flat_list, keep_idx = [], []

        for j, p in enumerate(batch_paths):
            try:
                if not os.path.exists(p):
                    continue
                sig, fs0 = read_sph_h5(p)            # [T,12], fs0
                sig = bandpass_fir(sig, fs0, 0.05, 47.0)
                sig, fs = resample_to_100hz(sig, fs0)
                sig = scale_to_unit(sig)
                sig = crop_center_5s(sig, fs)        # [500,12]
                flat = sig.reshape(-1).astype(np.float32)  # 500*12=6000
                if flat.shape[0] != 6000:
                    continue
                flat_list.append(flat)
                keep_idx.append(j)
            except Exception as e:
                # 出问题就跳过该样本
                # print(f"[WARN] skip {p}: {e}")
                continue

        if not flat_list:
            continue

        batch_flat = np.stack(flat_list, axis=0)      # [B,6000]
        emb = hubert_embed(batch_flat, model)         # [B,D]
        embs.append(emb)
        kept_Y.append(batch_Y[keep_idx])
        for j in keep_idx:
            kept_ids.append(ids[i+j])

        # 释放显存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(embs) == 0:
        raise RuntimeError("No embeddings produced. Check h5_dir & metadata.")

    X = np.concatenate(embs, axis=0)                 # [M,D]
    Y = np.concatenate(kept_Y, axis=0)               # [M,C]
    ids_arr = np.array(kept_ids, dtype=object)

    # 过滤（可选）：至少满足 min_keep 条正标签（或直接不过滤）
    if min_keep > 1:
        keep_mask = (Y.sum(axis=1) >= min_keep)
        X, Y, ids_arr = X[keep_mask], Y[keep_mask], ids_arr[keep_mask]

    # 保存
    np.savez_compressed(
        out_npz,
        X=X, Y=Y, ids=ids_arr,
        labels=np.array(TARGET_LABELS, dtype=object),
        hubert_id=np.array([HUBERT_ID], dtype=object),
        note=np.array(["SPH via HuBERT-ECG; 5s@100Hz; 12 leads flattened"], dtype=object)
    )
    print(f"Saved: {out_npz} | X={X.shape}, Y={Y.shape}, kept={len(ids_arr)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HuBERT-ECG embeddings for SPH dataset")
    ap.add_argument("--metadata", required=True, help="Path to metadata.csv (must include ECG_ID,AHA_Code)")
    ap.add_argument("--h5_dir",   required=True, help="Directory of <ECG_ID>.h5 files")
    ap.add_argument("--out",      default="./sph_hubert_embed.npz", help="Output NPZ path")
    ap.add_argument("--aha_col",  default="AHA_Code")
    ap.add_argument("--id_col",   default="ECG_ID")
    ap.add_argument("--bs",       type=int, default=64)
    ap.add_argument("--min-keep", type=int, default=1, help="Keep samples with >=min positives (1 keeps all)")
    ap.add_argument("--start",    type=int, default=0, help="Start index (inclusive)")
    ap.add_argument("--end",      type=int, default=None, help="End index (exclusive)")
    args = ap.parse_args()

    main(
        metadata_csv=args.metadata,
        h5_dir=args.h5_dir,
        out_npz=args.out,
        aha_col=args.aha_col,
        ecg_id_col=args.id_col,
        batch_size=args.bs,
        min_keep=args.min_keep,
        start=args.start,
        end=args.end
    )
