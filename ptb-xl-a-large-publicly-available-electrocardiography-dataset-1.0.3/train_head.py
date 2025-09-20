# train_head.py
import os
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class NpzEmbedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]


class SimpleClassifier(nn.Module):
    def __init__(self, hidden_dim: int, n_classes: int, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x): return self.net(x)


def train_one_epoch(model, loader, crit, optim, device):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_p.append(probs)
        all_y.append(y.numpy())
    y = np.concatenate(all_y, axis=0)
    p = np.concatenate(all_p, axis=0)
    # macro-F1、macro-AUROC
    try:
        f1 = f1_score(y, p>0.5, average="macro", zero_division=0)
    except Exception:
        f1 = 0.0
    try:
        auc = roc_auc_score(y, p, average="macro")
    except Exception:
        auc = 0.0
    return f1, auc


def main(npz_path: str, out_head: str = "./clf_head.pt", epochs=20, bs=128, lr=1e-3):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                 # [N,D]
    Y = data["Y"]                 # [N,C]
    labels = list(data["labels"])
    print("X:", X.shape, "Y:", Y.shape, "labels:", labels)

    Xtr, Xval, Ytr, Yval = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=(Y.sum(axis=1)>0))

    tr_ds = NpzEmbedDataset(Xtr, Ytr)
    va_ds = NpzEmbedDataset(Xval, Yval)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleClassifier(hidden_dim=X.shape[1], n_classes=Y.shape[1]).to(device)
    crit = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"f1": -1, "auc": -1}
    best_state = None

    for ep in range(1, epochs+1):
        loss = train_one_epoch(model, tr_ld, crit, optim, device)
        f1, auc = eval_model(model, va_ld, device)
        print(f"[Epoch {ep}] loss={loss:.4f}  f1={f1:.4f}  auc={auc:.4f}")
        if auc > best["auc"]:
            best = {"f1": f1, "auc": auc}
            best_state = model.state_dict().copy()

    # 保存最优
    if best_state is not None:
        torch.save(best_state, out_head)
        print(f"Saved best head to {out_head}  (val f1={best['f1']:.4f}, auc={best['auc']:.4f})")
    else:
        print("No best state saved.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to cached embeddings .npz")
    ap.add_argument("--out", default="./clf_head.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    main(args.npz, args.out, args.epochs, args.bs, args.lr)
