"""
lib/data.py
═══════════
Accès en lecture seule à results/ et artifacts/ (à la racine du projet, deux
niveaux au-dessus de demo/lib/). Ne dépend que de numpy/pandas, jamais de
torch/xgboost/sklearn : le tableau de bord doit rester utilisable même si
l'environnement d'entraînement n'est pas installé.

Chaque loader renvoie None si le fichier n'existe pas encore (rien n'est
inventé) : les pages affichent alors un état "en attente".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib.theme import MODEL_COLOR

ROOT      = Path(__file__).resolve().parents[2]
RESULTS   = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

MODELS = ["MLP", "LogReg", "XGBoost"]

# label affiché -> segment de nom de fichier
WB_ATTACKS  = {"FGSM": "fgsm", "PGD": "pgd", "C&W": "cw"}
BBS_ATTACKS = ["Square", "NES"]
BBD_ATTACKS = ["HSJA", "RayS"]

DEFENSES = [
    {"model": "MLP",     "name": "AT-FGSM",       "file": "mlp_at_fgsm.pt",
     "method": "Adversarial training (Madry-style), mix 50% exemples FGSM à chaque batch.",
     "threat": "white-box", "eps": 0.3},
    {"model": "MLP",     "name": "AT-PGD",        "file": "mlp_at_pgd.pt",
     "method": "Adversarial training, mix 50% exemples PGD (7 itérations) à chaque batch.",
     "threat": "white-box", "eps": 0.3},
    {"model": "LogReg",  "name": "Aug-FGSM",      "file": "logreg_aug_fgsm.pkl",
     "method": "Ré-entraînement (saga) sur train + exemples adverses FGSM.",
     "threat": "white-box", "eps": 0.3},
    {"model": "XGBoost", "name": "Aug-FGSM-Iter", "file": "xgb_iter_fgsm_r3.json",
     "method": "Augmentation itérative auto-générée, 3 rounds (génère → accumule → refit).",
     "threat": "white-box", "eps": 0.3},
    {"model": "MLP",     "name": "AT-Square",     "file": "mlp_at_square.pt",
     "method": "Augmentation itérative offline contre Square attack, 3 rounds.",
     "threat": "black-box", "eps": 0.3},
    {"model": "LogReg",  "name": "Aug-Square",    "file": "logreg_aug_square.pkl",
     "method": "Ré-entraînement sur train + exemples adverses Square attack.",
     "threat": "black-box", "eps": 0.3},
    {"model": "XGBoost", "name": "Aug-Square-Iter","file": "xgb_iter_square_r3.json",
     "method": "Augmentation itérative auto-générée contre Square, 3 rounds.",
     "threat": "black-box", "eps": 0.3},
]

DATASET_META = {
    "swat": {
        "name": "SWaT", "full": "Secure Water Treatment",
        "origin": "banc d'essai réel, iTrust Singapore",
        "note": "Données industrielles réelles, forte classe déséquilibrée (attaques rares).",
        "stages": [
            {"id": "P1", "label": "Admission & stockage brut", "tags": ["FIT101","LIT101","MV101","P101","P102"]},
            {"id": "P2", "label": "Dosage chimique",            "tags": ["AIT201","FIT201","MV201","P201","P202"]},
            {"id": "P3", "label": "Ultrafiltration",            "tags": ["DPIT301","FIT301","LIT301","MV301","P301"]},
            {"id": "P4", "label": "Déchloration UV",            "tags": ["AIT401","FIT401","LIT401","P401","UV401"]},
            {"id": "P5", "label": "Osmose inverse",             "tags": ["AIT501","FIT501","PIT501","P501","P502"]},
            {"id": "P6", "label": "Stockage & lavage",          "tags": ["FIT601","P601"]},
        ],
    },
    "batadal": {
        "name": "BATADAL", "full": "Battle of the Attack Detection Algorithms",
        "origin": "réseau simulé, challenge EPANET",
        "note": "Environnement simulé, attaques mieux définies mais moins de variabilité réelle.",
        "stages": None,
    },
}


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def feature_names(ds: str) -> list[str]:
    return _json(ARTIFACTS / ds / "feature_names.json") or []


@st.cache_data
def train_results(ds: str):
    return _json(RESULTS / ds / "train_results.json")


@st.cache_data
def whitebox_asr(ds: str) -> dict:
    """label -> {eps, status, asr:{model:val}} : utilise le run _tmp (non finalisé)
    si le JSON final n'existe pas encore, marqué "provisional"."""
    out = {}
    for label, key in WB_ATTACKS.items():
        d = _json(RESULTS / ds / f"whitebox_{key}_{ds}_eps0.1.json")
        if d:
            out[label] = {
                "eps": 0.1, "status": "final",
                "asr": {m: d.get(m, {}).get(label, {}).get("evasion_rate_median") for m in MODELS},
            }
            continue
        tmp = _csv(RESULTS / ds / f"whitebox_{key}_{ds}_eps0.1_tmp.csv")
        if tmp is not None and {"model", "asr"} <= set(tmp.columns):
            n_seeds = tmp["seed"].nunique() if "seed" in tmp.columns else None
            med = tmp.groupby("model")["asr"].median() * 100
            out[label] = {
                "eps": 0.1, "status": "provisional",
                "note": f"run non finalisé ({n_seeds} seed(s)), à relancer via whitebox_run.py",
                "asr": {m: (float(med[m]) if m in med.index else None) for m in MODELS},
            }
    return out


@st.cache_data
def blackbox_score_asr(ds: str):
    d = _json(RESULTS / ds / f"blackbox_score_{ds}_eps0.1.json")
    if not d:
        return None
    return {
        atk: {"asr": {m: d.get(m, {}).get(atk, {}).get("evasion_rate_median") for m in MODELS}}
        for atk in BBS_ATTACKS
    }


@st.cache_data
def blackbox_decision_asr(ds: str):
    d = _json(RESULTS / ds / f"blackbox_decision_{ds}_eps0.1.json")
    if not d:
        return None
    return {
        atk: {"asr": {m: d.get(m, {}).get(atk, {}).get("evasion_rate_median") for m in MODELS}}
        for atk in BBD_ATTACKS
    }


@st.cache_data
def transfer_asr(ds: str):
    return _json(RESULTS / ds / f"blackbox_transfer_{ds}_eps0.1.json")


@st.cache_data
def persample_whitebox(ds: str, attack_label: str):
    key = WB_ATTACKS.get(attack_label)
    if key is None:
        return None
    return _csv(RESULTS / ds / f"whitebox_persample_{key}_{ds}_eps0.1.csv")


@st.cache_data
def persample_blackbox_score(ds: str):
    return _csv(RESULTS / ds / f"blackbox_persample_score_{ds}_eps0.1.csv")


@st.cache_data
def persample_blackbox_decision(ds: str):
    return _csv(RESULTS / ds / f"blackbox_persample_decision_{ds}_eps0.1.csv")


@st.cache_data
def defense_results(ds: str):
    return _json(RESULTS / ds / "defense_results.json")


def defense_trained(ds: str, filename: str) -> bool:
    return (ARTIFACTS / ds / filename).exists()


@st.cache_data
def load_test_arrays(ds: str):
    """(X_test, y_test, timestamps_test) : chargement numpy pur, aucune dépendance ML."""
    adir = ARTIFACTS / ds
    X = np.load(adir / "X_test.npy")
    y = np.load(adir / "y_test.npy")
    ts = np.load(adir / "timestamps_test.npy", allow_pickle=True)
    return X, y, ts
