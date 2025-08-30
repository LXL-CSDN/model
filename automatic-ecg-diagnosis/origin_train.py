# origin_train.py
# PTB-XL official split (folds 1-8 train, 9 val, 10 test) + class weighting
# Trains only on: 2dAVB, 3dAVB, PAC, PVC, AFLT (customizable via --labels)

import os, json, math, argparse, h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                        ReduceLROnPlateau, CSVLogger, EarlyStopping)
from tensorflow.keras.utils import Sequence
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
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision: enabled (float16 compute, float32 vars)")
        except Exception as e:
            print("Mixed precision not enabled:", e)

# -----------------------
# Data loader (HDF5 on-the-fly)
# -----------------------
class H5ECGSequence(Sequence):
    def __init__(self, h5_path, idx_in_h5, y, batch_size=64, dataset_name="tracings", shuffle=True):
        self.h5_path = h5_path
        self.idx = np.asarray(idx_in_h5, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.n = len(self.idx)
        self.n_classes = self.y.shape[1]
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

        # h5py advanced indexing must be strictly increasing
        order = np.argsort(sel)
        sel_sorted = sel[order].astype(np.int64)
        with h5py.File(self.h5_path, "r") as hf:
            X_sorted = hf[self.dataset_name][sel_sorted, :, :]  # (B, 4096, 12)
        inv = np.argsort(order)
        X = X_sorted[inv, :, :].astype(np.float32)  # ensure float32

        Y = self.y[lo:hi]
        return X, Y

# -----------------------
# Utilities
# -----------------------
def infer_h5_indices(csv_df, h5_path):
    """
    Returns index array mapping csv rows -> HDF5 row indices.
    Prefer 'ecg_id' alignment via HDF5['ecg_id']; otherwise assume row order aligned.
    """
    with h5py.File(h5_path, "r") as hf:
        has_ids = "ecg_id" in hf.keys()
        if has_ids and "ecg_id" in csv_df.columns:
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

def compute_pos_weight(y, clip_max=50.0):
    """
    pos_weight = N_neg / N_pos per class, clipped for stability.
    """
    n = y.shape[0]
    pos = y.sum(axis=0)
    pos = np.clip(pos, 1.0, None)
    neg = n - pos
    pw = neg / pos
    if clip_max is not None:
        pw = np.minimum(pw, float(clip_max))
    return pw.astype(np.float32)

def build_weighted_bce(pos_weight_vec):
    """
    Weighted BCE on probabilities (sigmoid outputs).
    Cast to float32 to be safe under mixed precision.
    """
    pos_w = tf.constant(pos_weight_vec, dtype=tf.float32)

    @tf.function
    def weighted_bce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.constant(1e-7, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss_pos = - y_true * tf.math.log(y_pred)
        loss_neg = - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        loss = pos_w * loss_pos + loss_neg        # [B,C]
        return tf.reduce_mean(loss)
    return weighted_bce

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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor", type=str, default="val_AUCPR",
                        help="Metric to monitor for LR/ES/Checkpoint (e.g., val_AUCPR or val_AUCROC).")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading workers for fit().")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision.")
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
                              dataset_name=args.dataset_name, shuffle=True)
    valid_seq = H5ECGSequence(args.path_to_hdf5, idx_val, Y_val, batch_size=args.batch_size,
                              dataset_name=args.dataset_name, shuffle=False)
    test_seq  = H5ECGSequence(args.path_to_hdf5, idx_tst, Y_tst, batch_size=args.batch_size,
                              dataset_name=args.dataset_name, shuffle=False)

    # Class weights
    pos_weight = compute_pos_weight(Y_trn, clip_max=20.0)
    with open("class_pos_weight.json", "w") as f:
        json.dump({label_cols[i]: float(pos_weight[i]) for i in range(len(label_cols))}, f, indent=2)
    print("pos_weight:", {label_cols[i]: float(pos_weight[i]) for i in range(len(label_cols))})

    # Model
    model = get_model(train_seq.n_classes)  # expects sigmoid outputs for multi-label
    loss = build_weighted_bce(pos_weight)
    metrics = [
        tf.keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=train_seq.n_classes, name="AUCROC"),
        tf.keras.metrics.AUC(curve="PR",  multi_label=True, num_labels=train_seq.n_classes, name="AUCPR"),
    ]
    model.compile(optimizer=Adam(args.lr, clipnorm=1.0), loss=loss, metrics=metrics)

    # Callbacks (Keras 3 formats)
    callbacks = [
        ReduceLROnPlateau(monitor=args.monitor, factor=0.1, patience=7, min_lr=args.lr/100, mode="max"),
        EarlyStopping(monitor=args.monitor, patience=9, min_delta=1e-5, mode="max"),
        TensorBoard(log_dir="./logs", write_graph=False),
        CSVLogger("training.log", append=False),
        # Save last & best (full model .keras) + best weights (for compatibility)
        ModelCheckpoint("./backup_model_last.keras", save_best_only=False),
        ModelCheckpoint("./backup_model_best.keras", save_best_only=True, monitor=args.monitor, mode="max"),
        ModelCheckpoint("./backup_weights_best.weights.h5", save_best_only=True, save_weights_only=True,
                        monitor=args.monitor, mode="max"),
    ]

    # Train
    history = model.fit(
        train_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=valid_seq,
        verbose=1,
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
