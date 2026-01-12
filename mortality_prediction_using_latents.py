import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


latent_tags_csv_path = "latent_tags.csv"
physionet_ts_oc_ids_pkl_path = 'physionet2012_ts_oc_ids.pkl'


# =========================
# 1) Load & merge data
# =========================

def load_latents_and_outcomes(latent_tags_csv_path: str, physionet_ts_oc_ids_pkl_path: str) -> pd.DataFrame:
    # latent tags
    latent_df = pd.read_csv(latent_tags_csv_path)

    # outcomes (from your preprocess pickle)
    with open(physionet_ts_oc_ids_pkl_path, "rb") as f:
        ts, oc, ts_ids = pickle.load(f)

    # keep only what we need
    oc_small = oc[["ts_id", "in_hospital_mortality"]].copy()

    # merge
    latent_df["ts_id"] = latent_df["ts_id"].astype(str)
    oc_small["ts_id"] = oc_small["ts_id"].astype(str)
    df = latent_df.merge(oc_small, on="ts_id", how="inner")

    # sanity
    if df["in_hospital_mortality"].isna().any():
        df = df.dropna(subset=["in_hospital_mortality"])

    df["in_hospital_mortality"] = df["in_hospital_mortality"].astype(int)
    return df


def get_feature_columns(df: pd.DataFrame):
    # all columns except id + target
    return [c for c in df.columns if c not in ["ts_id", "in_hospital_mortality"]]


# =========================
# 2) Metrics helper
# =========================

def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "PosRate_true": float(y_true.mean()),
        "PosRate_pred": float(y_pred.mean()),
    }


# =========================
# 3) Baseline: Logistic Regression
# =========================

def train_logreg(X_train, y_train, X_val, y_val):
    # handle imbalance automatically
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    val_prob = clf.predict_proba(X_val)[:, 1]
    return clf, evaluate_probs(y_val, val_prob)


# =========================
# 4) DL: PyTorch MLP
# =========================

class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    ys = []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        pb = torch.sigmoid(logits).cpu().numpy()
        probs.append(pb)
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train_mlp(X_train, y_train, X_val, y_val, seed=42, epochs=50, batch_size=256, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TabDataset(X_train, y_train)
    val_ds = TabDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLP(in_dim=X_train.shape[1], hidden=32, dropout=0.1).to(device)

    # class imbalance: pos_weight = (#neg / #pos)
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auroc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # validate
        yv_true, yv_prob = predict_probs(model, val_loader, device)
        val_metrics = evaluate_probs(yv_true, yv_prob)

        if val_metrics["AUROC"] > best_val_auroc:
            best_val_auroc = val_metrics["AUROC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"[MLP] epoch {ep:03d} | val AUROC={val_metrics['AUROC']:.4f} | AUPRC={val_metrics['AUPRC']:.4f} | F1={val_metrics['F1']:.4f}")

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final val metrics (best model)
    yv_true, yv_prob = predict_probs(model, val_loader, device)
    return model, evaluate_probs(yv_true, yv_prob), device


# =========================
# 5) End-to-end runner
# =========================

def run_mortality_from_latents(
    latent_tags_csv_path: str,
    physionet_ts_oc_ids_pkl_path: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
):
    df = load_latents_and_outcomes(latent_tags_csv_path, physionet_ts_oc_ids_pkl_path)
    feat_cols = get_feature_columns(df)

    X = df[feat_cols].astype(float).values
    y = df["in_hospital_mortality"].values.astype(int)

    # split: train / temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # split temp into val/test (val_size is fraction of TRAIN, so convert)
    # Here: val is val_size of the original dataset portion that remains after train split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    # scale features (helpful for logreg & MLP)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"Data: N={len(df):,} | features={len(feat_cols)} | mortality_rate={y.mean():.3f}")
    print(f"Split: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")

    # --- Baseline ---
    logreg, val_metrics_lr = train_logreg(X_train_s, y_train, X_val_s, y_val)
    test_prob_lr = logreg.predict_proba(X_test_s)[:, 1]
    test_metrics_lr = evaluate_probs(y_test, test_prob_lr)

    print("\n[LogReg] VAL:", val_metrics_lr)
    print("[LogReg] TEST:", test_metrics_lr)

    # --- MLP ---
    mlp, val_metrics_mlp, device = train_mlp(X_train_s, y_train, X_val_s, y_val, seed=seed)
    test_loader = DataLoader(TabDataset(X_test_s, y_test), batch_size=512, shuffle=False)
    yt_true, yt_prob = predict_probs(mlp, test_loader, device)
    test_metrics_mlp = evaluate_probs(yt_true, yt_prob)

    print("\n[MLP]   VAL:", val_metrics_mlp)
    print("[MLP]   TEST:", test_metrics_mlp)

    return {
        "df": df,
        "feature_cols": feat_cols,
        "scaler": scaler,
        "logreg": logreg,
        "mlp": mlp,
        "metrics": {
            "logreg_val": val_metrics_lr,
            "logreg_test": test_metrics_lr,
            "mlp_val": val_metrics_mlp,
            "mlp_test": test_metrics_mlp,
        },
    }


out = run_mortality_from_latents(latent_tags_csv_path, physionet_ts_oc_ids_pkl_path)
