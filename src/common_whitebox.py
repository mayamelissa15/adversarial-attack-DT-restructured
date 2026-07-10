"""
common_whitebox.py
══════════════════
Helpers PARTAGÉS par le runner white-box unifié (whitebox_run.py).

Garantit que, pour un (dataset, seed, modèle) donné, l'eval set est
EXACTEMENT le même quelle que soit l'attaque (FGSM/PGD/C&W) : build_per_model_eval
utilise np.random.default_rng(seed), un RNG LOCAL indépendant de tout le reste.
→ les ASR restent comparables entre attaques, et le lancement en parallèle
  (un process par --attack) donne les mêmes résultats qu'un run --attack all.

Adaptations vs ancienne version :
  - Chemins via common.artifacts_dir / common.results_dir (fini les ~/<ds>/...).
  - Seuil = common.THRESHOLD (0.5) partout (fini le 0.45 codé en dur).
  - set_all_seeds ré-exporté depuis common (numpy + torch + cuda + cudnn).
  - --attack {fgsm,pgd,cw,all} ajouté au parser.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import joblib
from xgboost import XGBClassifier

from models import MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper
from common import THRESHOLD, artifacts_dir, results_dir, set_all_seeds  # noqa: F401
# set_all_seeds est ré-exporté pour que whitebox_run.py puisse l'importer d'ici.


# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

def build_arg_parser(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default="swat", choices=["swat", "batadal"],
                        help="Dataset cible")
    parser.add_argument("--attack", default="all",
                        choices=["fgsm", "pgd", "cw", "all"],
                        help="Attaque à lancer (remplace les 3 anciens runners)")
    parser.add_argument("--eps", default=0.1, type=float,
                        help="Epsilon L∞ (0.1 ou 0.3)")
    parser.add_argument("--n_runs", default=10, type=int, help="Nombre de seeds")
    parser.add_argument("--fast", action="store_true",
                        help="FAST_MODE : PGD 50x3 au lieu de 200x10, C&W 150 iters")
    parser.add_argument("--persample_n", default=50, type=int,
                        help="Nb max de samples gardés par (seed, modèle) dans le "
                             "CSV par-échantillon (analyse temporelle / timestamps)")
    return parser


# ══════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════

def setup_paths(dataset, eps):
    """Retourne (artifacts/<ds>, results/<ds>, tag) — dossiers créés au besoin."""
    save_dir = artifacts_dir(dataset)
    res_dir  = results_dir(dataset)
    tag = f"{dataset}_eps{eps}"
    return save_dir, res_dir, tag


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def eval_sizes(dataset):
    """SWaT : beaucoup d'attaques → 500. BATADAL : peu (≤44 en test) → 200 max."""
    eval_atk_size = 200 if dataset == "batadal" else 500
    eval_nrm_size = 500
    return eval_atk_size, eval_nrm_size


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES VICTIMES
# ══════════════════════════════════════════════════════════════

def load_victims(save_dir, device):
    """Charge X_test/y_test + les 3 modèles entraînés, enrobés dans leurs wrappers."""
    X_test = np.load(save_dir / "X_test.npy")
    y_test = np.load(save_dir / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(device)
    mlp_model.load_state_dict(torch.load(save_dir / "best_mlp.pt", map_location=device))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, device)

    logreg_w = LogRegWrapper(joblib.load(save_dir / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(save_dir / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    print(f"\n✓ Modèles chargés depuis {save_dir}")
    print(f"  X_test : {X_test.shape} — attaques : {int(y_test.sum())} / {len(y_test)}")
    return X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# EVAL SET PAR MODÈLE — logique de tirage à NE PAS modifier
# (sinon les attaques ne partagent plus le même eval set → ASR
#  non comparables entre FGSM / PGD / C&W)
# ══════════════════════════════════════════════════════════════

def build_per_model_eval(X_test, y_test, victim_w, seed,
                         eval_atk_size, eval_nrm_size, dataset):
    """
    Construit un eval set (normaux + attaques bien détectées) PROPRE à ce modèle
    et à ce seed, via un RNG LOCAL. Le seuil de détection est common.THRESHOLD.
    """
    rng = np.random.default_rng(seed)
    idx_normal = np.where(y_test == 0)[0]
    idx_attack = np.where(y_test == 1)[0]

    # On ne garde que les attaques que la victime détecte DÉJÀ (vrais positifs).
    preds_vic     = victim_w.predict(X_test[idx_attack], threshold=THRESHOLD)
    idx_attack_ok = idx_attack[preds_vic == 1]

    n_atk = min(eval_atk_size, len(idx_attack_ok))
    n_nrm = min(eval_nrm_size, len(idx_normal))
    if n_atk == 0:
        raise ValueError(
            f"Aucun vrai positif pour ce modèle sur {dataset} — vérifie le F1 baseline.")

    sel_n  = rng.choice(idx_normal,    size=n_nrm, replace=False)
    sel_a  = rng.choice(idx_attack_ok, size=n_atk, replace=n_atk > len(idx_attack_ok))
    idx_ev = np.concatenate([sel_n, sel_a])
    rng.shuffle(idx_ev)

    X_eval = X_test[idx_ev]
    y_eval = y_test[idx_ev]
    mask   = (y_eval == 1)
    X_atk  = X_eval[mask].astype(np.float32)
    y_atk  = y_eval[mask]
    return X_eval, y_eval, X_atk, y_atk, idx_ev


def load_timestamps(save_dir):
    """
    Charge timestamps_test.npy (désormais TOUJOURS produit par 00_train.py).
    Reste tolérant : si absent, désactive juste l'export par-échantillon.
    """
    path = save_dir / "timestamps_test.npy"
    if path.exists():
        return np.load(path, allow_pickle=True), True
    print(f"\n⚠ {path} introuvable — export par-échantillon désactivé "
          f"(relance 00_train.py --dataset <dataset>).")
    return None, False


VICTIMS_SPEC = [
    # (nom, is_logreg, is_xgb)
    ("MLP",     False, False),
    ("LogReg",  True,  False),
    ("XGBoost", False, True),
]