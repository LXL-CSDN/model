#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

import tensorflow as tf
from tensorflow import keras


# -------------------------------
# 配置：六分类标签映射（与模型输出顺序一致）
# -------------------------------
SIX_HEAD_ORDER = ['1dAVB', 'RBBB', 'LBBB', 'SBRAD', 'AFIB', 'STACH']
AHA_PRIMARY_TO_SIX = {
    '82':  '1dAVB',   # Prolonged PR interval
    '106': 'RBBB',    # Right bundle-branch block
    '104': 'LBBB',    # Left bundle-branch block
    '22':  'SBRAD',   # Sinus bradycardia
    '50':  'AFIB',    # Atrial fibrillation
    '21':  'STACH',   # Sinus tachycardia
}
SIX_INDEX = {k: i for i, k in enumerate(SIX_HEAD_ORDER)}


# -------------------------------
# 预处理（500 Hz -> 400 Hz；4096；12 导联）
# -------------------------------
class SPHPreprocessor:
    def __init__(self, target_fs=400, target_length=4096, target_channels=12, default_original_fs=500):
        self.target_fs = target_fs
        self.target_length = target_length
        self.target_channels = target_channels
        self.default_original_fs = default_original_fs

    def load_sph_ecg(self, h5_path: str) -> (np.ndarray, int):
        """
        读取 SPH .h5；期望 dataset 键为 'ecg'；形状可能为 (12, L) 或 (L, 12)
        返回 (ecg_array[L,12], original_fs)
        """
        with h5py.File(h5_path, 'r') as f:
            # 找到ECG数据
            key_candidates = ['ecg', 'ECG', 'signal', 'data']
            dset = None
            for k in key_candidates:
                if k in f:
                    dset = f[k][:]
                    break
            if dset is None:
                # 回退：取第一个 dataset
                for k in f.keys():
                    if isinstance(f[k], h5py.Dataset):
                        dset = f[k][:]
                        break
            if dset is None:
                raise ValueError("No ECG dataset found in H5.")

            ecg = np.array(dset)
            # 形状归一：希望 (L, 12)
            if ecg.ndim != 2:
                raise ValueError(f"Unexpected ECG ndim={ecg.ndim}")
            if ecg.shape[0] == 12 and ecg.shape[1] != 12:
                ecg = ecg.T  # (12, L) -> (L, 12)
            elif ecg.shape[1] == 12:
                pass  # already (L, 12)
            else:
                raise ValueError(f"ECG shape not compatible: {ecg.shape}")

            # 采样率来源：attrs 或默认 500
            fs = self.default_original_fs
            for attr_key in ['sampling_rate', 'fs', 'SamplingRate', 'sample_rate']:
                if attr_key in f.attrs:
                    try:
                        fs = int(f.attrs[attr_key])
                        break
                    except Exception:
                        pass

        return ecg.astype(np.float32), int(fs)

    def resample_to_target(self, ecg_array: np.ndarray, original_fs: int) -> np.ndarray:
        if original_fs == self.target_fs:
            return ecg_array
        n_samples = ecg_array.shape[0]
        new_len = int(round(n_samples * self.target_fs / float(original_fs)))
        out = np.zeros((new_len, ecg_array.shape[1]), dtype=np.float32)
        for ch in range(ecg_array.shape[1]):
            out[:, ch] = signal.resample(ecg_array[:, ch], new_len)
        return out

    def normalize_length(self, ecg_array: np.ndarray) -> np.ndarray:
        L = ecg_array.shape[0]
        if L > self.target_length:
            return ecg_array[:self.target_length, :]
        elif L < self.target_length:
            pad = self.target_length - L
            return np.pad(ecg_array, ((0, pad), (0, 0)), mode='constant', constant_values=0)
        return ecg_array

    def validate(self, ecg_array: np.ndarray) -> bool:
        if np.isnan(ecg_array).any() or np.isinf(ecg_array).any():
            return False
        if (ecg_array == 0).all():
            return False
        max_amp = float(np.abs(ecg_array).max())
        if max_amp < 0.01 or max_amp > 50.0:
            # 按 mV 粗略范围
            return False
        if ecg_array.shape != (self.target_length, self.target_channels):
            return False
        return True

    def preprocess(self, h5_path: str) -> (np.ndarray, bool):
        try:
            ecg, fs0 = self.load_sph_ecg(h5_path)
            ecg = self.resample_to_target(ecg, fs0)
            ecg = self.normalize_length(ecg)
            ok = self.validate(ecg)
            return ecg.astype(np.float32), ok
        except Exception as e:
            print(f"[Warn] {os.path.basename(h5_path)} failed: {e}")
            return np.zeros((self.target_length, self.target_channels), dtype=np.float32), False


# -------------------------------
# 标签：AHA_Code → 六分类多标签向量
# -------------------------------
def parse_label_6(code_str: str) -> np.ndarray:
    """
    输入 AHA_Code，如 '21+308;104;50'
    仅取每条语句 '+' 前的主语句代码，并映射到六类向量
    """
    y = np.zeros(len(SIX_HEAD_ORDER), dtype=np.float32)
    if code_str is None:
        return y
    s = str(code_str).strip()
    if not s:
        return y

    for stmt in s.split(';'):
        stmt = stmt.strip()
        if not stmt:
            continue
        primary = stmt.split('+')[0].strip()
        cls = AHA_PRIMARY_TO_SIX.get(primary)
        if cls is not None:
            y[SIX_INDEX[cls]] = 1.0
    return y


# -------------------------------
# 指标
# -------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold=0.5) -> dict:
    y_bin = (y_prob >= threshold).astype(np.float32)

    metrics = {}
    metrics['subset_accuracy'] = float(accuracy_score(y_true, y_bin))
    metrics['hamming_score']  = float((y_true == y_bin).mean())

    for avg in ['samples', 'macro', 'micro', 'weighted']:
        p = float(precision_score(y_true, y_bin, average=avg, zero_division=0))
        r = float(recall_score(y_true, y_bin, average=avg, zero_division=0))
        f = float(f1_score(y_true, y_bin, average=avg, zero_division=0))
        metrics[f'precision_{avg}'] = p
        metrics[f'recall_{avg}'] = r
        metrics[f'f1_{avg}'] = f

    # AUC（仅对有正样本的类别）
    try:
        valid = y_true.sum(axis=0) > 0
        if valid.any():
            metrics['auc_macro']    = float(roc_auc_score(y_true[:, valid], y_prob[:, valid], average='macro'))
            metrics['auc_weighted'] = float(roc_auc_score(y_true[:, valid], y_prob[:, valid], average='weighted'))
        else:
            metrics['auc_macro'] = None
            metrics['auc_weighted'] = None
    except Exception as e:
        print(f"[Warn] AUC failed: {e}")
        metrics['auc_macro'] = None
        metrics['auc_weighted'] = None

    return metrics, y_bin


def per_class_report(y_true: np.ndarray, y_bin: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(SIX_HEAD_ORDER):
        yt = y_true[:, i]
        yp = y_bin[:, i]
        support = int(yt.sum())
        if support == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        rows.append({
            'Label': name,
            'Support': support,
            'Precision': float(precision_score(yt, yp, zero_division=0)),
            'Recall': float(recall_score(yt, yp, zero_division=0)),
            'F1-Score': float(f1_score(yt, yp, zero_division=0)),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        })
    return pd.DataFrame(rows)


# -------------------------------
# 绘图（matplotlib）
# -------------------------------
def plot_overall(metrics: dict, out_png: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Overall bars
    ax1 = axes[0, 0]
    overall = {
        'Subset Acc': metrics['subset_accuracy'],
        'Hamming':    metrics['hamming_score'],
        'F1 (Samples)': metrics['f1_samples'],
        'F1 (Macro)':   metrics['f1_macro'],
        'F1 (Micro)':   metrics['f1_micro'],
        'F1 (Weighted)':metrics['f1_weighted'],
    }
    ax1.bar(list(overall.keys()), list(overall.values()))
    ax1.set_ylim(0, 1)
    ax1.set_title('Overall Metrics')
    ax1.grid(axis='y', alpha=0.3)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(30)

    # Precision/Recall/F1 compare
    ax2 = axes[0, 1]
    groups = {
        'Macro':   [metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro']],
        'Micro':   [metrics['precision_micro'], metrics['recall_micro'], metrics['f1_micro']],
        'Weighted':[metrics['precision_weighted'], metrics['recall_weighted'], metrics['f1_weighted']],
    }
    idx = np.arange(3)
    w = 0.25
    for i, (k, vals) in enumerate(groups.items()):
        ax2.bar(idx + i*w, vals, w, label=k)
    ax2.set_xticks(idx + w)
    ax2.set_xticklabels(['Precision','Recall','F1'])
    ax2.set_ylim(0, 1)
    ax2.set_title('Precision / Recall / F1')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # AUC
    ax3 = axes[1, 0]
    if metrics.get('auc_macro') is not None:
        ax3.bar(['AUC (Macro)','AUC (Weighted)'], [metrics['auc_macro'], metrics['auc_weighted']])
        ax3.set_ylim(0, 1)
        ax3.set_title('ROC AUC')
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'AUC Not Available', ha='center', va='center')
        ax3.set_axis_off()

    # Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    rows = [
        ['Subset Accuracy', f"{metrics['subset_accuracy']:.4f}"],
        ['Hamming Score',   f"{metrics['hamming_score']:.4f}"],
        ['F1 (Samples)',    f"{metrics['f1_samples']:.4f}"],
        ['F1 (Macro)',      f"{metrics['f1_macro']:.4f}"],
        ['F1 (Micro)',      f"{metrics['f1_micro']:.4f}"],
        ['F1 (Weighted)',   f"{metrics['f1_weighted']:.4f}"],
        ['Precision (Macro)', f"{metrics['precision_macro']:.4f}"],
        ['Recall (Macro)',    f"{metrics['recall_macro']:.4f}"],
    ]
    if metrics.get('auc_macro') is not None:
        rows.append(['AUC (Macro)', f"{metrics['auc_macro']:.4f}"])
    tbl = ax4.table(cellText=rows, colLabels=['Metric','Value'], cellLoc='left', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_per_class(per_class_df: pd.DataFrame, out_png: str, top_n: int = 20):
    if per_class_df.empty:
        return
    df = per_class_df.sort_values('Support', ascending=False).head(top_n)
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # PRF bars
    ax1 = axes[0]
    x = np.arange(len(df))
    w = 0.25
    ax1.bar(x - w, df['Precision'].values, w, label='Precision')
    ax1.bar(x,     df['Recall'].values,    w, label='Recall')
    ax1.bar(x + w, df['F1-Score'].values,  w, label='F1-Score')
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Label'].tolist(), rotation=30, ha='right')
    ax1.set_title(f'Per-Class Performance (Top {top_n} by Support)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Support bars
    ax2 = axes[1]
    ax2.bar(x, df['Support'].values)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Label'].tolist(), rotation=30, ha='right')
    ax2.set_title('Support Distribution')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------------------
# 数据迭代（流式）
# -------------------------------
def iter_dataset(metadata: pd.DataFrame, data_dir: str, preproc: SPHPreprocessor,
                 batch_size: int = 64, sample_size: int = None):
    meta = metadata.copy()
    if sample_size is not None and sample_size < len(meta):
        meta = meta.sample(n=sample_size, random_state=42)

    X_batch, Y_batch, IDs = [], [], []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Preparing"):
        ecg_id = str(row['ECG_ID'])
        h5_path = os.path.join(data_dir, f"{ecg_id}.h5")
        if not os.path.exists(h5_path):
            continue

        ecg, ok = preproc.preprocess(h5_path)
        if not ok:
            continue

        y = parse_label_6(row['AHA_Code'])
        X_batch.append(ecg.astype(np.float32))
        Y_batch.append(y.astype(np.float32))
        IDs.append(ecg_id)

        if len(X_batch) == batch_size:
            yield np.stack(X_batch), np.stack(Y_batch), IDs
            X_batch, Y_batch, IDs = [], [], []

    if X_batch:
        yield np.stack(X_batch), np.stack(Y_batch), IDs


# -------------------------------
# 预测接口（自动对齐输入形状）
# -------------------------------
def predict_batches(model, X: np.ndarray, batch_size: int):
    exp = model.input_shape  # e.g. (None, 4096, 12) or (None, 12, 4096)
    Xin = X
    if isinstance(exp, (list, tuple)) and len(exp) >= 3:
        # 统一期望：(N, 4096, 12) 或 (N, 12, 4096)
        if exp[-1] == 12 and X.shape[1:] == (12, 4096):
            Xin = np.transpose(X, (0, 2, 1))  # -> (N,4096,12)
        elif exp[-1] == 4096 and X.shape[1:] == (4096, 12):
            Xin = np.transpose(X, (0, 2, 1))  # -> (N,12,4096)
    return model.predict(Xin, batch_size=batch_size, verbose=0)


# -------------------------------
# 主流程
# -------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 读 metadata
    meta = pd.read_csv(args.metadata)
    need_cols = {'ECG_ID', 'AHA_Code'}
    if not need_cols.issubset(set(meta.columns)):
        raise ValueError(f"metadata.csv must contain columns: {need_cols}")

    # 加载模型
    model = keras.models.load_model(args.model_path)
    print("Model loaded. input_shape:", model.input_shape, " output_shape:", model.output_shape)

    preproc = SPHPreprocessor(target_fs=400, target_length=4096, target_channels=12, default_original_fs=500)

    all_true, all_prob, all_ids = [], [], []

    for X, Y, IDs in iter_dataset(meta, args.data_dir, preproc,
                                  batch_size=args.batch_size, sample_size=args.sample_size):
        y_prob = predict_batches(model, X, batch_size=args.batch_size)
        # 若模型输出维度不为 (B, 6)，尝试截取或映射
        if y_prob.ndim == 2 and y_prob.shape[1] >= 6:
            y_prob = y_prob[:, :6]
        elif y_prob.ndim == 1 and y_prob.shape[0] == X.shape[0]:
            # 罕见：单输出 -> 重复到 6 列（占位），以免后续报错
            y_prob = np.repeat(y_prob[:, None], 6, axis=1)
        else:
            # 尽量提示
            print(f"[Warn] Unexpected model output shape {y_prob.shape}, expecting (*,6).")

        all_true.append(Y)
        all_prob.append(y_prob)
        all_ids.extend(IDs)

    if not all_true:
        raise RuntimeError("No valid samples processed.")

    y_true = np.vstack(all_true)
    y_prob = np.vstack(all_prob)
    metrics, y_bin = compute_metrics(y_true, y_prob, threshold=0.5)
    per_cls = per_class_report(y_true, y_bin)

    # 保存
    with open(os.path.join(args.output_dir, "overall_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    per_cls.to_csv(os.path.join(args.output_dir, "per_class_metrics.csv"), index=False)

    # 绘图
    plot_overall(metrics, os.path.join(args.output_dir, "overall_metrics.png"))
    plot_per_class(per_cls, os.path.join(args.output_dir, "per_class_performance.png"), top_n=20)

    # 简要打印
    print("\n===== Overall =====")
    for k in ['subset_accuracy','hamming_score','f1_samples','f1_macro','f1_micro','f1_weighted','auc_macro','auc_weighted']:
        v = metrics.get(k)
        print(f"{k:>16}: {v if v is None else f'{v:.4f}'}")
    print("\nTop classes by support:")
    if not per_cls.empty:
        print(per_cls.sort_values('Support', ascending=False).head(10).to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a 6-head DNN on SPH dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model.hdf5")
    parser.add_argument("--data_dir",   type=str, required=True, help="Directory containing <ECG_ID>.h5 files")
    parser.add_argument("--metadata",   type=str, required=True, help="Path to metadata.csv (with ECG_ID, AHA_Code)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for preprocessing/prediction")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional sample size for smoke test")
    args = parser.parse_args()
    main(args)
