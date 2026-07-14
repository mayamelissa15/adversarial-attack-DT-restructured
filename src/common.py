"""
common.py
═════════
Infrastructure PARTAGÉE par tout le projet.

Ce module ne définit AUCUN modèle (voir models.py). Il rassemble ce que
plusieurs scripts utilisent en commun :

  1. Constantes globales     → SEED, THRESHOLD  (une seule source de vérité)
  2. Chemins du projet       → data/, artifacts/, results/
  3. set_all_seeds()         → reproductibilité totale (numpy / torch / cuda)
  4. load_dataset()          → chargement propre de SWaT ET BATADAL
  5. Utilitaires d'éval      → build_eval_set, eval_attack, eval_attack_persample
                               (déplacés depuis l'ancien models.py)

Aucune dépendance vers models.py → pas de dépendance circulaire (les fonctions
d'éval reçoivent un wrapper EN ARGUMENT, elles n'importent aucune classe).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


# ══════════════════════════════════════════════════════════════
# 1. CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════

SEED = 42

# Seuil de décision UNIQUE, appliqué à TOUS les modèles, PARTOUT.
# (Corrige l'incohérence historique : 0.5 à l'entraînement vs 0.45 aux attaques.)
THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════
# 2. CHEMINS DU PROJET
# ══════════════════════════════════════════════════════════════
# __file__ = these_adv/src/common.py  →  .parent.parent = these_adv/
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR   = ROOT / "results"


def artifacts_dir(dataset: str) -> Path:
    """Retourne (en le créant au besoin) artifacts/<dataset>/."""
    d = ARTIFACTS_DIR / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(dataset: str) -> Path:
    """Retourne (en le créant au besoin) results/<dataset>/."""
    d = RESULTS_DIR / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def subsample_for_substitute(X, y, max_n=100_000, seed=SEED):
    """
    Sous-échantillonne (X, y) avant l'entraînement d'un substitut black-box.

    Sur SWaT (922 700 lignes), entraîner 3 substituts par seed sur le train
    set complet est le vrai goulot du pipeline transfer (30 entraînements
    complets sur ~1M lignes, en CPU). Un sous-échantillon aléatoire de
    max_n lignes garde le ratio de classes (tirage uniforme) et suffit très
    largement à entraîner un substitut représentatif — un attaquant réel n'a
    de toute façon jamais accès au train set complet de la victime.
    """
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), max_n, replace=False)
    return X[idx], y[idx]


# ══════════════════════════════════════════════════════════════
# 3. REPRODUCTIBILITÉ
# ══════════════════════════════════════════════════════════════

def set_all_seeds(seed: int = SEED) -> None:
    """
    Fixe TOUTES les sources d'aléa pour rendre l'entraînement reproductible.

    Pourquoi autant de lignes ? Chaque bibliothèque a son propre générateur :
      - random           : shuffles Python purs
      - numpy            : splits sklearn, tirages divers
      - torch (CPU)      : init des poids, dropout, shuffle du DataLoader
      - torch.cuda       : mêmes opérations mais côté GPU
      - cudnn            : certains kernels GPU sont non déterministes par défaut ;
                           on force le mode déterministe (un peu plus lent, mais
                           indispensable pour que 2 runs donnent le MÊME résultat).

    Les appels .cuda sont inoffensifs s'il n'y a pas de GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════
# 4. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════
#
# Chaque loader renvoie un TRIPLET homogène :
#   X   : DataFrame (N, D) des features numériques uniquement
#   y   : ndarray  (N,)    d'entiers 0 (Normal) / 1 (Attack)
#   ts  : ndarray  (N,)    de timestamps (datetime64), aligné position/position
#
# C'est ce format commun qui permet à 00_train.py d'être UNIQUE : il ne sait
# pas quel dataset il manipule, il reçoit toujours (X, y, ts).
# ──────────────────────────────────────────────────────────────

# SWaT : le label peut apparaître sous plusieurs formes (dont le fameux typo
# " A ttack" avec espaces). On les mappe toutes explicitement.
_SWAT_LABEL_MAP = {"Normal": 0, "Attack": 1, "A ttack": 1}


def _load_swat():
    """Charge SWaT depuis data/raw/swat/merged.csv."""
    path = DATA_DIR / "swat" / "merged.csv"
    # skipinitialspace=True : supprime les espaces après les virgules
    # (les colonnes SWaT type " MV101" et les valeurs " 2.44").
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()      # nettoie aussi les noms de colonnes

    # ---- Label ----
    raw = df["Normal/Attack"].astype(str).str.strip()
    y = raw.map(_SWAT_LABEL_MAP)
    if y.isna().any():
        raise ValueError(f"Labels SWaT inconnus : {raw[y.isna()].unique()}")
    y = y.astype(int).to_numpy()

    # ---- Timestamp : "28/12/2015 10:00:00 AM" (jour/mois/année 12h + AM/PM) ----
    ts = pd.to_datetime(
        df["Timestamp"].astype(str).str.strip(),
        format="%d/%m/%Y %I:%M:%S %p",
    ).to_numpy()

    # ---- Features : tout sauf timestamp et label ----
    X = df.drop(columns=["Timestamp", "Normal/Attack"])
    return X, y, ts


def _load_batadal():
    """
    Charge BATADAL depuis data/raw/batadal/.

    Construction du jeu supervisé (décision méthodologique) :
      - Normal (0) : TOUT dataset03 (100 % fonctionnement sain, ATT_FLAG == 0)
      - Attack (1) : SEULEMENT les lignes de dataset04 avec ATT_FLAG == 1
      - on JETTE les ATT_FLAG == -999 (étiquettes cachées du challenge, non fiables)
    """
    d = DATA_DIR / "batadal"
    df3 = pd.read_csv(d / "BATADAL_dataset03.csv", skipinitialspace=True)
    df4 = pd.read_csv(d / "BATADAL_dataset04.csv", skipinitialspace=True)
    for df in (df3, df4):
        df.columns = df.columns.str.strip()   # CRUCIAL : ds03 et ds04 diffèrent
                                               # par les espaces (" L_T1" vs "L_T1")

    # dataset03 → tout normal
    df3 = df3.copy()
    df3["label"] = 0

    # dataset04 → uniquement les attaques confirmées, le reste (-999) est jeté
    df4 = df4[df4["ATT_FLAG"] == 1].copy()
    df4["label"] = 1

    # Colonnes de features = tout sauf DATETIME / ATT_FLAG / label
    feature_cols = [c for c in df3.columns if c not in ("DATETIME", "ATT_FLAG", "label")]

    combined = pd.concat([df3, df4], ignore_index=True)

    y  = combined["label"].to_numpy().astype(int)
    # ---- Timestamp BATADAL : "06/01/14 00" (jour/mois/année 2 chiffres + heure) ----
    ts = pd.to_datetime(
        combined["DATETIME"].astype(str).str.strip(),
        format="%d/%m/%y %H",
    ).to_numpy()

    X = combined[feature_cols]
    return X, y, ts


def load_dataset(name: str):
    """Dispatcher : 'swat' ou 'batadal' → (X_df, y, timestamps)."""
    name = name.lower()
    if name == "swat":
        return _load_swat()
    if name == "batadal":
        return _load_batadal()
    raise ValueError(f"dataset inconnu : {name!r} (attendu 'swat' ou 'batadal')")


# ══════════════════════════════════════════════════════════════
# 5. UTILITAIRES D'ÉVALUATION (déplacés depuis l'ancien models.py)
# ══════════════════════════════════════════════════════════════

def build_eval_set(X_test, y_test, wrapper, threshold=THRESHOLD):
    """
    Construit le jeu d'évaluation d'attaque = les VRAIS POSITIFS clean.
    On ne cherche à évader QUE les attaques déjà correctement détectées.
    """
    y_pred = wrapper.predict(X_test, threshold)
    mask = (y_test == 1) & (y_pred == 1)
    return X_test[mask].astype(np.float32), y_test[mask]


def eval_attack(wrapper, X_full, y_full, X_adv, attack_name, model_name,
                threshold=THRESHOLD):
    """
    Évalue une attaque adversariale (résumé agrégé).

    ASR = fraction des vrais positifs clean qui deviennent classés "normaux"
    après perturbation.
    """
    y_pred_clean = wrapper.predict(X_full, threshold=threshold)
    tp_mask      = (y_full == 1) & (y_pred_clean == 1)
    n_tp         = int(tp_mask.sum())

    if n_tp == 0:
        return {
            "attack": attack_name, "model": model_name, "asr": 0.0,
            "f1_clean": f1_score(y_full, y_pred_clean, zero_division=0),
            "f1_adv": f1_score(y_full, y_pred_clean, zero_division=0),
            "rec_adv": recall_score(y_full, y_pred_clean, zero_division=0),
            "prec_adv": precision_score(y_full, y_pred_clean, zero_division=0),
            "linf": 0.0, "linf_mean": 0.0, "linf_max": 0.0,
            "margin_clean_mean": np.nan, "margin_adv_mean": np.nan,
            "margin_adv_median": np.nan, "margin_adv_min": np.nan,
            "margin_drop_mean": np.nan, "n_tp": 0,
        }

    if len(X_adv) != n_tp:
        raise ValueError(
            f"X_adv contient {len(X_adv)} samples, mais il y a {n_tp} TP clean "
            f"dans X_full. X_adv doit être généré sur X_full[tp_mask]."
        )

    X_eval = X_full.copy()
    X_eval[tp_mask] = X_adv
    y_pred_adv = wrapper.predict(X_eval, threshold=threshold)

    asr = float((y_pred_adv[tp_mask] == 0).mean())

    diff            = np.abs(X_adv - X_full[tp_mask])
    linf_per_sample = np.max(diff, axis=1)

    # Marge = logit − logit_seuil : >0 encore détecté attaque, <0 évadé.
    threshold_logit = float(np.log(threshold / (1.0 - threshold)))
    clean_margin = wrapper.logits_np(X_full[tp_mask]) - threshold_logit
    adv_margin   = wrapper.logits_np(X_adv)           - threshold_logit

    return {
        "attack":    attack_name,
        "model":     model_name,
        "asr":       asr,
        "f1_clean":  f1_score(y_full, y_pred_clean, zero_division=0),
        "f1_adv":    f1_score(y_full, y_pred_adv,   zero_division=0),
        "rec_adv":   recall_score(y_full, y_pred_adv,    zero_division=0),
        "prec_adv":  precision_score(y_full, y_pred_adv, zero_division=0),
        "linf":      float(np.max(linf_per_sample)),
        "linf_mean": float(np.mean(linf_per_sample)),
        "linf_max":  float(np.max(linf_per_sample)),
        "margin_clean_mean": float(np.mean(clean_margin)),
        "margin_adv_mean":   float(np.mean(adv_margin)),
        "margin_adv_median": float(np.median(adv_margin)),
        "margin_adv_min":    float(np.min(adv_margin)),
        "margin_drop_mean":  float(np.mean(clean_margin - adv_margin)),
        "n_tp": n_tp,
    }


def eval_attack_persample(wrapper, X_full, y_full, X_adv, attack_name, model_name,
                          timestamps_full, threshold=THRESHOLD, max_samples=50, seed=0):
    """
    Version par-échantillon (pour l'analyse temporelle) : un dict par attaque,
    sous-échantillonné à max_samples de façon reproductible.

    timestamps_full DOIT être aligné position/position avec X_full/y_full.
    """
    y_pred_clean = wrapper.predict(X_full, threshold=threshold)
    tp_mask      = (y_full == 1) & (y_pred_clean == 1)
    n_tp         = int(tp_mask.sum())
    if n_tp == 0:
        return []
    if len(X_adv) != n_tp:
        raise ValueError(
            f"X_adv contient {len(X_adv)} samples, mais il y a {n_tp} TP clean."
        )

    X_eval = X_full.copy()
    X_eval[tp_mask] = X_adv
    y_pred_adv = wrapper.predict(X_eval, threshold=threshold)
    success_per_sample = (y_pred_adv[tp_mask] == 0)

    threshold_logit       = float(np.log(threshold / (1.0 - threshold)))
    adv_margin_per_sample = wrapper.logits_np(X_adv) - threshold_logit
    linf_per_sample       = np.max(np.abs(X_adv - X_full[tp_mask]), axis=1)
    timestamps_tp         = np.asarray(timestamps_full)[tp_mask]

    rng      = np.random.default_rng(seed)
    n_keep   = min(max_samples, n_tp)
    keep_idx = rng.choice(n_tp, size=n_keep, replace=False)

    return [{
        "timestamp":  timestamps_tp[i],
        "model":      model_name,
        "attack":     attack_name,
        "seed":       seed,
        "success":    bool(success_per_sample[i]),
        "margin_adv": float(adv_margin_per_sample[i]),
        "linf":       float(linf_per_sample[i]),
    } for i in keep_idx]