"""
00_train.py — Entraînement UNIFIÉ (SWaT + BATADAL)
═══════════════════════════════════════════════════

Un seul script pour les deux datasets, piloté par --dataset.

    python src/00_train.py --dataset swat
    python src/00_train.py --dataset batadal

Ce que ce script garantit (tes 4 corrections méthodo) :
  1. VRAI validation set  → early stopping du MLP ET de XGBoost sur X_val ;
                            X_test n'est JAMAIS regardé pendant l'entraînement.
  2. SEEDS fixés          → set_all_seeds() (numpy/torch/cuda) + DataLoader seedé.
  3. SEUIL unique         → common.THRESHOLD appliqué à predict_proba() pour les
                            TROIS modèles (fini le .predict() implicite à 0.5).
  4. COMPARABILITÉ        → un seul flux (X, y, ts), splits stratifiés ; les
                            timestamps voyagent DANS le split → timestamps_*.npy
                            restent alignés (rend inutile 00_extract_timestamps.py).

Sorties :
  artifacts/<dataset>/ : X_{train,val,test}.npy, y_*.npy, timestamps_*.npy,
                         scaler.pkl, feature_names.json, best_mlp.pt,
                         logreg.pkl, xgb.json
  results/<dataset>/   : train_results.json (pour le dashboard)
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score,
                             precision_score, recall_score)
from xgboost import XGBClassifier
import joblib

from common import (SEED, THRESHOLD, set_all_seeds, load_dataset,
                    artifacts_dir, results_dir)
from models import MLP


# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Entraînement unifié SWaT / BATADAL")
    p.add_argument("--dataset", required=True, choices=["swat", "batadal"])
    p.add_argument("--test-size",  type=float, default=0.2,
                   help="fraction du dataset réservée au TEST (défaut 0.2)")
    p.add_argument("--val-size",   type=float, default=0.2,
                   help="fraction du TRAINVAL réservée à la VALIDATION (défaut 0.2)")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--patience",   type=int,   default=5)
    p.add_argument("--batch-size", type=int,   default=2048)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--n-seeds",    type=int,   default=1,
                   help="nombre de seeds essayés pour le MLP ; on garde celui qui "
                        "maximise le F1 sur le VALIDATION set (défaut 1 = comportement "
                        "historique, seed=SEED uniquement)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def metrics_at_threshold(y_true, proba, threshold=THRESHOLD):
    """Métriques calculées au SEUIL UNIQUE (proba >= threshold)."""
    y_pred = (proba >= threshold).astype(int)
    return {
        "clean_accuracy": round(float((y_pred == y_true).mean()) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }, y_pred


def class_counts(name, y):
    print(f"  {name:5s} → Normal {int((y==0).sum()):>6d} | Attack {int((y==1).sum()):>5d}")


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_all_seeds(SEED)

    ds     = args.dataset
    A      = artifacts_dir(ds)
    R      = results_dir(ds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Dataset: {ds} | device: {device} | seuil unique: {THRESHOLD} ===")

    # ── 1. Chargement (X_df, y, timestamps) ────────────────────
    X_df, y, ts = load_dataset(ds)
    feature_names = list(X_df.columns)
    X_all = X_df.to_numpy(dtype=np.float64)
    print(f"\nChargé : {X_all.shape[0]} lignes, {X_all.shape[1]} features")
    class_counts("total", y)

    # ── 2. Split TRAINVAL / TEST (stratifié, timestamps embarqués) ──
    X_tv, X_test, y_tv, y_test, ts_tv, ts_test = train_test_split(
        X_all, y, ts, test_size=args.test_size, stratify=y, random_state=SEED
    )
    # ── 3. Split TRAIN / VAL depuis TRAINVAL (stratifié) ──
    X_train, X_val, y_train, y_val, ts_train, ts_val = train_test_split(
        X_tv, y_tv, ts_tv, test_size=args.val_size, stratify=y_tv, random_state=SEED
    )
    print("\nRépartition des splits :")
    class_counts("train", y_train)
    class_counts("val",   y_val)
    class_counts("test",  y_test)

    # ── 4. Imputation NaN — moyennes du TRAIN uniquement (pas de fuite) ──
    col_means = np.nanmean(X_train, axis=0)
    def impute(M):
        bad = np.isnan(M)
        if bad.any():
            M[bad] = np.take(col_means, np.where(bad)[1])
        return M
    X_train, X_val, X_test = impute(X_train), impute(X_val), impute(X_test)

    # ── 5. Standardisation — scaler ajusté sur le TRAIN uniquement ──
    scaler  = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ── Sauvegarde des splits + métadonnées ──
    np.save(A / "X_train.npy", X_train); np.save(A / "y_train.npy", y_train)
    np.save(A / "X_val.npy",   X_val);   np.save(A / "y_val.npy",   y_val)
    np.save(A / "X_test.npy",  X_test);  np.save(A / "y_test.npy",  y_test)
    np.save(A / "timestamps_train.npy", ts_train)
    np.save(A / "timestamps_val.npy",   ts_val)
    np.save(A / "timestamps_test.npy",  ts_test)
    joblib.dump(scaler, A / "scaler.pkl")
    with open(A / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print("\nSplits + scaler + métadonnées sauvegardés ✓")

    # Poids de classe (déséquilibre) — calculés sur le TRAIN
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = int((y_train == 0).sum())
    imbalance = n_neg / n_pos

    # ══════════════════════════════════════════════════════════
    # MLP — entraînement pour UN seed donné (early stopping sur VAL)
    # ══════════════════════════════════════════════════════════
    def train_mlp_one_seed(seed):
        set_all_seeds(seed)
        model = MLP(X_train.shape[1]).to(device)

        Xtr_t  = torch.tensor(X_train)
        ytr_t  = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        Xval_t = torch.tensor(X_val).to(device)

        gen = torch.Generator(); gen.manual_seed(seed)   # shuffle reproductible
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                            batch_size=args.batch_size, shuffle=True, generator=gen)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([imbalance], dtype=torch.float32, device=device)
        )
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_f1, no_improve, best_state = -1.0, 0, None
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()

            # Early stopping sur la VALIDATION (jamais sur le test)
            model.eval()
            with torch.no_grad():
                proba_val = torch.sigmoid(model(Xval_t)).cpu().numpy().flatten()
            f1_val = f1_score(y_val, (proba_val >= THRESHOLD).astype(int), zero_division=0)

            if f1_val > best_f1:
                best_f1, no_improve = f1_val, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

        model.load_state_dict(best_state)
        return model, best_f1

    # ══════════════════════════════════════════════════════════
    # MLP — sélection multi-seed (n_seeds=1 → comportement historique)
    # ══════════════════════════════════════════════════════════
    print(f"\n=== MLP ({args.n_seeds} seed{'s' if args.n_seeds > 1 else ''}) ===")
    seeds_to_try = [SEED + i for i in range(args.n_seeds)]
    best_seed, model, best_val_f1 = None, None, -1.0
    for seed in seeds_to_try:
        candidate, val_f1 = train_mlp_one_seed(seed)
        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1, model, best_seed = val_f1, candidate, seed
            marker = "  ← meilleur jusqu'ici"
        if args.n_seeds > 1:
            print(f"  seed={seed:4d}  val F1={val_f1:.4f}{marker}")

    if args.n_seeds > 1:
        print(f"  → seed retenu = {best_seed} (val F1 = {best_val_f1:.4f})")

    torch.save(model.state_dict(), A / "best_mlp.pt")

    model.eval()
    with torch.no_grad():
        proba_mlp = torch.sigmoid(
            model(torch.tensor(X_test).to(device))
        ).cpu().numpy().flatten()

    # ══════════════════════════════════════════════════════════
    # LOGISTIC REGRESSION
    # ══════════════════════════════════════════════════════════
    print("\n=== LogReg ===")
    logreg = LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                class_weight="balanced", random_state=SEED)
    logreg.fit(X_train, y_train)
    joblib.dump(logreg, A / "logreg.pkl")
    proba_lr = logreg.predict_proba(X_test)[:, 1]

    # ══════════════════════════════════════════════════════════
    # XGBOOST
    # ══════════════════════════════════════════════════════════
    print("\n=== XGBoost ===")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=imbalance, eval_metric="logloss",
        early_stopping_rounds=20, tree_method="hist",
        device=device, random_state=SEED, verbosity=0,
    )
    # early stopping sur la VALIDATION (le test reste vierge)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb.save_model(str(A / "xgb.json"))
    proba_xgb = xgb.predict_proba(X_test)[:, 1]

    # ══════════════════════════════════════════════════════════
    # ÉVALUATION AU SEUIL UNIQUE + RAPPORTS
    # ══════════════════════════════════════════════════════════
    results = {}
    for name, proba in [("MLP", proba_mlp), ("LogReg", proba_lr), ("XGBoost", proba_xgb)]:
        m, y_pred = metrics_at_threshold(y_test, proba, THRESHOLD)
        results[name] = m
        print(f"\n=== {name} (seuil {THRESHOLD}) ===")
        print(classification_report(y_test, y_pred,
                                    target_names=["Normal", "Attack"], zero_division=0))

    results["_meta"] = {
        "dataset": ds, "threshold": THRESHOLD, "seed": SEED,
        "mlp_n_seeds": args.n_seeds, "mlp_best_seed": best_seed,
        "mlp_best_val_f1": round(best_val_f1, 4),
    }
    with open(R / "train_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Terminé. Artefacts → {A}")
    print(f"✓ Résultats JSON     → {R / 'train_results.json'}")


if __name__ == "__main__":
    main()