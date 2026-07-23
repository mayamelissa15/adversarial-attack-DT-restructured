"""
threshold_tuning.py — Recherche du seuil optimal F1 par modèle (analyse seule)
═══════════════════════════════════════════════════════════════════════════

Ne réentraîne RIEN. Charge les artefacts déjà produits par 00-train.py
(best_mlp.pt, logreg.pkl, xgb.json + X_val/y_val/X_test/y_test) et cherche,
POUR CHAQUE MODÈLE INDÉPENDAMMENT, le seuil qui maximise le F1 sur le
validation set (jamais sur le test). Puis applique ce seuil au test set et
compare au seuil unique 0.5 actuellement utilisé partout.

Usage :
    python src/threshold_tuning.py --dataset swat
    python src/threshold_tuning.py --dataset batadal
"""

import argparse

import numpy as np
import torch
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from common import THRESHOLD, artifacts_dir
from models import MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper


def parse_args():
    p = argparse.ArgumentParser(description="Recherche du seuil F1-optimal par modèle")
    p.add_argument("--dataset", required=True, choices=["swat", "batadal"])
    return p.parse_args()


def best_threshold(y_val, proba_val):
    """Balaye 999 seuils sur le validation set, retourne celui qui maximise F1."""
    grid = np.linspace(0.001, 0.999, 999)
    f1s = [f1_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])


def metrics_at(y_true, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    return {
        "acc":  round(float((y_pred == y_true).mean()) * 100, 2),
        "f1":   round(f1_score(y_true, y_pred, zero_division=0), 4),
        "prec": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "rec":  round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def main():
    args = parse_args()
    ds = args.dataset
    A = artifacts_dir(ds)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_val   = np.load(A / "X_val.npy");   y_val   = np.load(A / "y_val.npy")
    X_test  = np.load(A / "X_test.npy");  y_test  = np.load(A / "y_test.npy")

    mlp = MLP(X_val.shape[1]).to(device)
    mlp.load_state_dict(torch.load(A / "best_mlp.pt", map_location=device))
    mlp.eval()
    logreg = joblib.load(A / "logreg.pkl")
    xgb = XGBClassifier()
    xgb.load_model(str(A / "xgb.json"))

    wrappers = {
        "MLP":     MLPWrapper(mlp, device),
        "LogReg":  LogRegWrapper(logreg),
        "XGBoost": XGBoostWrapper(xgb),
    }

    print(f"\n=== {ds.upper()} — seuil unique actuel = {THRESHOLD} ===\n")
    print(f"{'Modèle':8s} {'seuil*':>7s} | {'F1@0.5':>7s} {'F1@seuil*':>9s} | "
          f"{'Prec@0.5':>9s} {'Prec@seuil*':>11s} | {'Rec@0.5':>8s} {'Rec@seuil*':>10s} | "
          f"{'Acc@0.5':>8s} {'Acc@seuil*':>10s}")

    for name, w in wrappers.items():
        proba_val  = w.predict_proba(X_val)
        proba_test = w.predict_proba(X_test)

        t_star, f1_val_star = best_threshold(y_val, proba_val)

        m_05 = metrics_at(y_test, proba_test, THRESHOLD)
        m_st = metrics_at(y_test, proba_test, t_star)

        print(f"{name:8s} {t_star:7.3f} | {m_05['f1']:7.4f} {m_st['f1']:9.4f} | "
              f"{m_05['prec']:9.4f} {m_st['prec']:11.4f} | "
              f"{m_05['rec']:8.4f} {m_st['rec']:10.4f} | "
              f"{m_05['acc']:8.2f} {m_st['acc']:10.2f}")

    print("\n(seuil* choisi en maximisant le F1 sur le VALIDATION set ; "
          "métriques @seuil* évaluées sur le TEST set, jamais l'inverse)")


if __name__ == "__main__":
    main()
