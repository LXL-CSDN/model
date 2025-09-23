# embed_ptbxl.py
import os
import json
import ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

import torch
from transformers import AutoModel
from scipy import signal


# ====== 配置 ======
TARGET_LABELS = ["1AVB","CRBBB","CLBBB","AFIB","STACH","2AVB","3AVB","PAC","PVC","AFLT"]  # 可按需修改
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUBERT_ID = "Edoardo-BS/hubert-ecg-base"  # 也可换 large
TRUST_REMOTE_CODE = True                  # 允许自定义代码（HF 仓库需要）


# ====== 预处理：与论文一致 ======
def bandpass_fir(sig: np.ndarray, fs: int, low=0.05, high=47.0) -> np.ndarray:
    """FIR 带通 + filtfilt；sig: [T, C]"""
    x = np.atleast_2d(sig).astype(np.float32)
    T = x.shape[0]
    nyq = fs / 2
    high = min(high, nyq * 0.98)

    # 自适应 taps，避免 padlen 报错
    target_taps = int(3 * fs)          # ~3秒窗
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
    """
    Reads the signal from the PTB-XL records path (without extension) at records100, returning a flattened 1D (length=6000), fs=100
    """
    sig, meta = wfdb.rdsamp(rec_path_no_ext)  # sig: [T, 12]
    fs = int(meta["fs"])
    bp = bandpass_fir(sig, fs, 0.05, 47.0)
    sig100, fs100 = resample_to_100hz(bp, fs)
    unit = scale_to_unit(sig100)
    s5 = crop_center_5s(unit, fs100)  # [500, 12]
    flat = s5.reshape(-1).astype(np.float32)  # 500*12=6000
    return flat, 100


# ====== 标签解析（PTB-XL 的 scp_codes 是 dict 字符串）======
def parse_scp_dict(s: str) -> Dict[str, float]:
    """
    'scp_codes' 字段是形如 "{'AFLT': 100.0, 'ABQRS': 0.0, 'AFIB': 0.0}" 的字符串。
    解析后把所有 value 强制为 float。
    """
    d = {}
    if not isinstance(s, str) or not s.strip():
        return d
    try:
        d = ast.literal_eval(s)
    except Exception:
        try:
            d = json.loads(s.replace("'", '"'))
        except Exception:
            return {}
    out = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out


def multilabel_row(scp: Dict[str, float], target_labels: List[str], min_score: float = 50.0) -> np.ndarray:
    """
    仅当 label 在 target_labels 中且 scp[label] > min_score 才置 1，否则为 0
    如需 >=50 也算阳性，将判断改为 >= min_score
    """
    return np.array(
        [1.0 if (k in scp and float(scp[k]) > min_score) else 0.0 for k in target_labels],
        dtype=np.float32
    )


# ====== 嵌入提取 ======
@torch.no_grad()
def hubert_embed(batch_flat: np.ndarray, model: AutoModel) -> torch.Tensor:
    """
    batch_flat: [B, 6000] float32 in [-1,1]
    返回: [B, D] 的 pooled embedding（mean pool）
    """
    x = torch.tensor(batch_flat, dtype=torch.float32, device=DEVICE)
    # 直接喂给 input_values（仓库自定义代码通常支持）
    out = model(input_values=x)
    # 兼容不同输出键
    if hasattr(out, "last_hidden_state"):
        hs = out.last_hidden_state  # [B, T, D]
    else:
        hs = out[0]
    emb = hs.mean(dim=1)  # [B, D]
    return emb  # 不 detach 也行，这里 no_grad


def main_embed(
    ptbxl_csv: str,
    records100_dir: str,
    out_npz: str,
    folds_for_train: List[int] = [1,2,3,4,5,6,7,8],
    batch_size: int = 64,
    min_score: float = 50.0
):
    df = pd.read_csv(ptbxl_csv)
    df = df[df['filename_lr'].notna()].copy()
    df['scp_codes_dict'] = df['scp_codes'].apply(parse_scp_dict)

    df_train = df[df['strat_fold'].isin(folds_for_train)].reset_index(drop=True)

    # 预取目标标签
    y_list = []
    paths = []
    for _, row in df_train.iterrows():
        y_list.append(multilabel_row(row['scp_codes_dict'], TARGET_LABELS, min_score=min_score))
        # 构造 rdsamp 需要的无后缀路径
        # e.g. records100/00000/00001/00001
        rec_path = os.path.join(records100_dir, row['filename_lr'])
        rec_path = os.path.splitext(rec_path)[0]
        paths.append(rec_path)

    y = np.stack(y_list, axis=0)  # [N, C]
    N = len(paths)

    # 加载 HuBERT-ECG（只 forward）
    model = AutoModel.from_pretrained(HUBERT_ID, trust_remote_code=TRUST_REMOTE_CODE).to(DEVICE)
    model.eval()

    # 批量预处理 + 提取 embedding
    embs = []
    ids  = []
    kept_y = []  # 与 ids 对齐的标签
    for i in tqdm(range(0, N, batch_size), desc="Embedding"):
        batch_paths = paths[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        batch_flat = []
        ok_idx = []
        for j, p in enumerate(batch_paths):
            try:
                flat, fs = preprocess_wfdb_record(p)
                assert fs == 100 and flat.shape[0] == 6000
                batch_flat.append(flat)
                ok_idx.append(j)
                ids.append(p)
            except Exception as e:
                print(f"[WARN] skip {p}: {e}")

        if not batch_flat:
            continue

        batch_flat = np.stack(batch_flat, axis=0)        # [B,6000]
        emb = hubert_embed(batch_flat, model)            # [B,D]
        embs.append(emb.cpu().numpy())
        kept_y.append(batch_y[ok_idx])

    if len(embs) == 0:
        raise RuntimeError("No embeddings produced. Check paths and preprocessing.")

    X = np.concatenate(embs, axis=0)                 
    Y = np.concatenate(kept_y, axis=0)               

    np.savez_compressed(
        out_npz,
        X=X, Y=Y, ids=np.array(ids, dtype=object),
        labels=np.array(TARGET_LABELS, dtype=object),
        hubert_id=np.array([HUBERT_ID], dtype=object),
        min_score=np.array([min_score], dtype=np.float32)
    )
    print(f"Saved embeddings to: {out_npz}  | X={X.shape}, Y={Y.shape}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl-csv", required=True, help="Path to ptbxl_database.csv")
    ap.add_argument("--records100", required=True, help="Path to records100 folder")
    ap.add_argument("--out", default="./ptbxl_train_embed.npz", help="Output NPZ")
    ap.add_argument("--folds", type=int, nargs="+", default=[1,2,3,4,5,6,7,8])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--min-score", type=float, default=50.0,
                    help="仅当 scp_codes[label] > min-score 记为阳性（默认 50.0）")
    args = ap.parse_args()

    main_embed(args.ptbxl_csv, args.records100, args.out, args.folds, args.bs, min_score=args.min_score)
