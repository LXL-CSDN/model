# origin_train.py
# 官方折 (1-8 train / 9 val / 10 test) + logits损失 + AdamW + L2/Dropout 正则 + 稳定训练

import os, json, math, argparse, h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.optimizers import AdamW
from keras.callbacks import (ModelCheckpoint, TensorBoard,
                             ReduceLROnPlateau, CSVLogger, EarlyStopping, TerminateOnNaN)
from keras.utils import Sequence
from model import get_model

DEFAULT_LABELS = ["2dAVB", "3dAVB", "PAC", "PVC", "AFLT"]

# -----------------------
# GPU & mixed precision
# -----------------------
def setup_acceleration(enable_mixed_precision=True):
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print("GPUs:", gpus)
    if enable_mixed_precision:
        try:
            from keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision: enabled (float16 compute, float32 vars)")
        except Exception as e:
            print("Mixed precision not enabled:", e)

# -----------------------
# Metrics that accept logits (apply sigmoid internally)
# -----------------------
class AUCFromLogits(keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.sigmoid(tf.cast(y_pred, tf.float32))
        y_true = tf.cast(y_true, tf.float32)
        return super().update_state(y_true, y_pred, sample_weight)

def make_auc_metrics(n_classes):
    return [
        AUCFromLogits(curve="ROC",  multi_label=True, num_labels=n_classes, name="AUCROC"),
        AUCFromLogits(curve="PR",   multi_label=True, num_labels=n_classes, name="AUCPR"),
    ]

# -----------------------
# Data loader (HDF5 on-the-fly, persistent handle)
# -----------------------
class H5ECGSequence(Sequence):
    def __init__(self, h5_path, idx_in_h5, y, batch_size=64, dataset_name="tracings", shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.h5_path = h5_path
        self.idx = np.asarray(idx_in_h5, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.augment = augment and shuffle  # 只对训练集生效
        self.n = len(self.idx)
        self.n_classes = self.y.shape[1]
        # 持久 HDF5 句柄
        self.hf = h5py.File(self.h5_path, "r")
        self.Xds = self.hf[self.dataset_name]
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            p = np.random.permutation(self.n)
            self.idx, self.y = self.idx[p], self.y[p]

    def __getitem__(self, i):
        lo = i * self.batch_size
        hi = min((i + 1) * self.batch_size, self.n)
        sel = self.idx[lo:hi]  # may be unsorted

        # h5py 高级索引必须升序
        order = np.argsort(sel)
        sel_sorted = sel[order].astype(np.int64)
        X_sorted = self.Xds[sel_sorted, :, :]  # (B, 4096, 12)
        inv = np.argsort(order)
        X = X_sorted[inv, :, :].astype(np.float32)
        Y = self.y[lo:hi]

        if self.augment:
            X = self._augment(X)
        return X, Y

    def _augment(self, X):
        # 轻量增强：幅度缩放 ±10%、微噪声、轻微平移、time mask
        B, L, C = X.shape
        # 1) scale
        scale = np.random.uniform(0.9, 1.1, size=(B, 1, 1)).astype(np.float32)
        X *= scale
        # 2) noise
        X += np.random.normal(0, 0.003, size=X.shape).astype(np.float32)
        # 3) shift
        shift = np.random.randint(-80, 81)
        if shift != 0:
            X = np.roll(X, shift, axis=1)
        # 4) time mask
        if np.random.rand() < 0.3:
            w = np.random.randint(int(0.01*L), int(0.015*L))
            s = np.random.randint(0, L - w)
            X[:, s:s+w, :] = 0.0
        return X

    def __del__(self):
        try:
            self.hf.close()
        except:
            pass

# -----------------------
# Utilities
# -----------------------
def infer_h5_indices(csv_df, h5_path):
    with h5py.File(h5_path, "r") as hf:
        has_ids = "ecg_id" in hf and "ecg_id" in csv_df
        if has_ids:
            h5_ids = hf["ecg_id"][:].astype(int)
            csv_ids = csv_df["ecg_id"].astype(int).values
            id2idx = {int(e): int(i) for i, e in enumerate(h5_ids)}
            idx = np.array([id2idx[int(e)] for e in csv_ids], dtype=np.int64)
            return idx
        else:
            n_h5 = hf["tracings"].shape[0]
            if len(csv_df) > n_h5:
                raise ValueError("CSV has more rows than HDF5; cannot align without ecg_id.")
            return np.arange(len(csv_df), dtype=np.int64)

def split_official(df_all):
    trn = df_all[df_all["strat_fold"].isin([1,2,3,4,5,6,7,8])].reset_index(drop=True)
    val = df_all[df_all["strat_fold"] == 9].reset_index(drop=True)
    tst = df_all[df_all["strat_fold"] == 10].reset_index(drop=True)
    return trn, val, tst

def select_labels(df, label_cols):
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in CSV: {missing}")
    return df[label_cols].astype(np.float32).values

def compute_pos_weight(y, clip_max=10.0):
    n = y.shape[0]
    pos = y.sum(axis=0)
    pos = np.clip(pos, 1.0, None)
    neg = n - pos
    pw = neg / pos
    if clip_max is not None:
        pw = np.minimum(pw, float(clip_max))
    return pw.astype(np.float32)

def build_weighted_bce_logits(pos_weight_vec):
    pos_w = tf.constant(pos_weight_vec, dtype=tf.float32)
    @tf.function
    def loss_fn(y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=logits, pos_weight=pos_w)
        # tf.nn.weighted_cross_entropy_with_logits 已做逐元素，取均值即可
        return tf.reduce_mean(loss)
    return loss_fn

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train with PTB-XL official split (folds 1-8/9/10).")
    parser.add_argument("path_to_hdf5", type=str, help="Path to HDF5 containing tracings (and optional ecg_id).")
    parser.add_argument("path_to_csv", type=str, help="Path to CSV with labels (must include strat_fold).")
    parser.add_argument("--dataset_name", type=str, default="tracings", help="HDF5 dataset name (default: tracings).")
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS),
                        help="Comma-separated label names to train on.")
    parser.add_argument("--batch_size", type=int, default=48)  # 略降，提升泛化+缓解显存
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor", type=str, default="val_AUCPR",
                        help="Metric to monitor for LR/ES/Checkpoint (e.g., val_AUCPR or val_AUCROC).")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--augment", action="store_true", help="Enable light data augmentation for training.")
    parser.add_argument("--resume_from", type=str, default="", help="Path to a saved .keras model to resume from")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Resume initial epoch")
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Acceleration setup
    setup_acceleration(enable_mixed_precision=not args.no_mixed_precision)

    # Read CSV and check columns
    df = pd.read_csv(args.path_to_csv)
    if "strat_fold" not in df.columns:
        raise ValueError("CSV must include 'strat_fold' column for official split.")
    label_cols = [c.strip() for c in args.labels.split(",") if c.strip()]

    # Official split
    df_trn, df_val, df_tst = split_official(df)

    # Labels
    Y_trn = select_labels(df_trn, label_cols)
    Y_val = select_labels(df_val, label_cols)
    Y_tst = select_labels(df_tst, label_cols)

    # Map CSV rows to HDF5 indices
    idx_trn = infer_h5_indices(df_trn, args.path_to_hdf5)
    idx_val = infer_h5_indices(df_val, args.path_to_hdf5)
    idx_tst = infer_h5_indices(df_tst, args.path_to_hdf5)

    # Sequences
    train_seq = H5ECGSequence(args.path_to_hdf5, idx_trn, Y_trn, batch_size=args.batch_size,
                              dataset_name=args.dataset_name, shuffle=True, augment=args.augment)
    valid_seq = H5ECGSequence(args.path_to_hdf5, idx_val, Y_val, batch_size=args.batch_size,
                              dataset_name=args.dataset_name, shuffle=False, augment=False)
    test_seq  = H5ECGSequence(args.path_to_hdf5, idx_tst, Y_tst, batch_size=args.batch_size,
                              dataset_name=args.dataset_name, shuffle=False, augment=False)

    # Class weights
    pos_weight = compute_pos_weight(Y_trn, clip_max=10.0)
    with open("class_pos_weight.json", "w") as f:
        json.dump({label_cols[i]: float(pos_weight[i]) for i in range(len(label_cols))}, f, indent=2)
    print("pos_weight:", {label_cols[i]: float(pos_weight[i]) for i in range(len(label_cols))})

    # Model & loss & metrics
    loss = build_weighted_bce_logits(pos_weight)
    if args.resume_from and os.path.exists(args.resume_from):
        model = keras.models.load_model(args.resume_from, custom_objects={"loss_fn": loss, "AUCFromLogits": AUCFromLogits})
        print(f"Resumed from {args.resume_from}")
        # 重新编译（以防优化器策略变化）
        metrics = make_auc_metrics(train_seq.n_classes)
        opt = AdamW(learning_rate=args.lr, weight_decay=1e-4)
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
    else:
        model = get_model(train_seq.n_classes)
        metrics = make_auc_metrics(train_seq.n_classes)
        opt = AdamW(learning_rate=args.lr, weight_decay=1e-4)
        # 为稳健再加梯度裁剪
        opt.clipnorm = 1.0
        model.compile(optimizer=opt, loss=loss, metrics=metrics)

    # Callbacks (Keras 3 formats)
    callbacks = [
        ReduceLROnPlateau(monitor=args.monitor, factor=0.2, patience=3, min_lr=1e-6, mode="max"),
        EarlyStopping(monitor=args.monitor, patience=5, min_delta=1e-5, mode="max", restore_best_weights=True),
        TerminateOnNaN(),
        TensorBoard(log_dir="./logs", write_graph=False),
        CSVLogger("training.log", append=False),
        ModelCheckpoint("./backup_model_last.keras", save_best_only=False),
        ModelCheckpoint("./backup_model_best.keras", save_best_only=True, monitor=args.monitor, mode="max"),
        ModelCheckpoint("./backup_weights_best.weights.h5", save_best_only=True, save_weights_only=True,
                        monitor=args.monitor, mode="max"),
    ]

    # Train
    history = model.fit(
        train_seq,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        callbacks=callbacks,
        validation_data=valid_seq,
        verbose=1
    )

    # Evaluate on test fold (fold 10)
    print("Evaluating on official test fold (10)...")
    eval_vals = model.evaluate(test_seq, verbose=1, return_dict=True)
    with open("test_metrics.json", "w") as f:
        json.dump(eval_vals, f, indent=2)
    print("Test metrics:", eval_vals)

    # Save final model & weights
    model.save("./final_model.keras")
    model.save_weights("./final_weights.weights.h5")
    # Save label order for downstream
    with open("label_order.json", "w") as f:
        json.dump(label_cols, f, indent=2)

if __name__ == "__main__":
    main()
