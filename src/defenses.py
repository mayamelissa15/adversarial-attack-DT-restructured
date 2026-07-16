"""
defenses.py — Entraînement des modèles DÉFENDUS
════════════════════════════════════════════════

Produit, pour un dataset donné, sept modèles durcis : quatre contre des
attaques WHITE-BOX (gradient exact/numérique), trois contre une attaque
BLACK-BOX (Square, à requêtes) :

  - MLP AT-FGSM            → mlp_at_fgsm.pt          (adversarial training, Madry-style)
  - MLP AT-PGD              → mlp_at_pgd.pt
  - LogReg Aug-FGSM          → logreg_aug_fgsm.pkl     (augmentation adverse)
  - XGBoost Aug-FGSM-Iter    → xgb_iter_fgsm_r{R}.json (augmentation itérative auto)
  - MLP AT-Square (bb)       → mlp_at_square.pt        (augmentation itérative, offline)
  - LogReg Aug-Square (bb)   → logreg_aug_square.pkl   (augmentation adverse)
  - XGBoost Aug-Square-Iter  → xgb_iter_square_r{R}.json (augmentation itérative auto)

Usage :
    python src/defenses.py --dataset swat
    python src/defenses.py --dataset batadal --eps-at 0.3

Choix de design (thèse) : les ATTAQUES sont menées à eps=0.1, mais les
DÉFENSES sont entraînées à eps=0.3 (--eps-at), y compris les variantes
black-box. Un modèle durci contre des perturbations plus larges généralise
mieux vers les plus petites.

Pourquoi les défenses black-box sont OFFLINE (pas per-batch comme AT-FGSM/PGD) :
Square est une attaque à requêtes (des centaines par exemple, batchées sur
tout le batch). La regénérer à CHAQUE batch de CHAQUE epoch serait bien trop
coûteux. On adopte donc, pour le MLP aussi, le même schéma "génère → accumule
→ réentraîne" déjà utilisé pour XGBoost (--bb-rounds rounds, cf.
augment_xgboost_iterative), au lieu du online per-batch des AT-FGSM/AT-PGD.

Corrections vs ancienne version :
  1. eps d'entraînement 0.1 → 0.3 (design assumé).
  2. Early stopping sur le VRAI X_val.npy (avant : X_train[:5000], = données
     déjà vues → mesure inutile).
  3. Seuil = common.THRESHOLD (0.5). Chemins via common.artifacts_dir. --dataset.
  4. BUG corrigé : le MLP s'entraînait dropout ÉTEINT (fgsm/pgd passent le
     modèle en .eval() ; on repasse en .train() avant le pas d'optimisation).
  5. Seeds fixés (numpy/torch/cuda) + DataLoader seedé → reproductible.

Si un fichier de sortie existe déjà, il est rechargé sans réentraîner (reprise).
"""

import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from common import SEED, THRESHOLD, set_all_seeds, artifacts_dir
from models import MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper
from whitebox import fgsm_mlp, pgd_mlp, fgsm_logreg, pgd_logreg, fgsm_xgb, pgd_xgb
from blackbox import square_attack


# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Entraînement des modèles défendus")
    p.add_argument("--dataset", required=True, choices=["swat", "batadal"])
    p.add_argument("--eps-at", type=float, default=0.3,
                   help="Epsilon d'adversarial training (design : 0.3)")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--patience",   type=int,   default=7)
    p.add_argument("--batch-size", type=int,   default=2048)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--mix-ratio",  type=float, default=0.5,
                   help="Fraction du batch remplacée par des exemples adverses")
    p.add_argument("--xgb-rounds", type=int,   default=3,
                   help="Nb de rounds d'augmentation itérative XGBoost")
    p.add_argument("--bb-rounds", type=int,   default=3,
                   help="Nb de rounds d'augmentation itérative black-box (MLP/XGBoost)")
    p.add_argument("--bb-max-queries", type=int, default=300,
                   help="Budget de requêtes Square par round (coût de la génération)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES ARTEFACTS
# ══════════════════════════════════════════════════════════════

def load_artifacts(save_dir, device):
    X_train = np.load(save_dir / "X_train.npy")
    y_train = np.load(save_dir / "y_train.npy")
    X_val   = np.load(save_dir / "X_val.npy")
    y_val   = np.load(save_dir / "y_val.npy")

    mlp_model = MLP(input_size=X_train.shape[1]).to(device)
    mlp_model.load_state_dict(torch.load(save_dir / "best_mlp.pt", map_location=device))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, device)

    logreg_w = LogRegWrapper(joblib.load(save_dir / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(save_dir / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    print(f"✓ Artefacts chargés depuis {save_dir}")
    print(f"  X_train {X_train.shape} | X_val {X_val.shape}")
    return X_train, y_train, X_val, y_val, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# DÉFENSE 1 & 2 — ADVERSARIAL TRAINING MLP (FGSM ou PGD)
#
# À chaque batch : 1) génère X_adv sur le modèle courant (eval),
#                  2) mélange mix_ratio clean/adv,
#                  3) entraîne (train) sur le mix.
# ══════════════════════════════════════════════════════════════

def adversarial_train_mlp(X_train, y_train, X_val, y_val, input_size,
                          save_dir, device, attack, eps,
                          epochs, patience, batch_size, lr, mix_ratio):
    fpath = save_dir / f"mlp_at_{attack}.pt"
    if fpath.exists():
        print(f"    {fpath.name} déjà présent → chargement direct")
        model = MLP(input_size=input_size).to(device)
        model.load_state_dict(torch.load(fpath, map_location=device))
        model.eval()
        return MLPWrapper(model, device)

    set_all_seeds(SEED)
    model = MLP(input_size=input_size).to(device)

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = int((y_train == 0).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    gen = torch.Generator(); gen.manual_seed(SEED)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True, generator=gen)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_f1, no_improve, best_state = -1.0, 0, None
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # 1) Génération des exemples adverses sur le modèle COURANT.
            #    (fgsm_mlp / pgd_mlp mettent le modèle en .eval() en interne.)
            tmp_w = MLPWrapper(model, device)
            xb_np = xb.detach().cpu().numpy()
            yb_np = yb.detach().cpu().numpy().flatten().astype(int)
            if attack == "fgsm":
                xb_adv_np = fgsm_mlp(tmp_w, xb_np, yb_np, eps=eps)
            else:
                xb_adv_np = pgd_mlp(tmp_w, xb_np, yb_np, eps=eps,
                                    iters=7, restarts=1, verbose=False)
            xb_adv = torch.tensor(xb_adv_np, dtype=torch.float32, device=device)

            # 2) Mix clean / adv
            n_adv   = int(len(xb) * mix_ratio)
            idx_adv = torch.randperm(len(xb), device=device)[:n_adv]
            xb_mix  = xb.clone()
            xb_mix[idx_adv] = xb_adv[idx_adv]

            # 3) Pas d'entraînement — ⚠ RETOUR EN MODE TRAIN (dropout actif).
            model.train()
            optimizer.zero_grad()
            criterion(model(xb_mix), yb).backward()
            optimizer.step()

        # Early stopping sur le VRAI validation set
        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
        f1 = f1_score(y_val, (proba >= THRESHOLD).astype(int), zero_division=0)

        if f1 > best_f1:
            best_f1, no_improve = f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop epoch {epoch+1} — best val F1 {best_f1:.4f}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d} | val F1 {f1:.4f} | best {best_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), fpath)
    print(f"    Sauvegardé : {fpath.name}  (best val F1 {best_f1:.4f})")
    return MLPWrapper(model, device)


# ══════════════════════════════════════════════════════════════
# DÉFENSE 5 — AUGMENTATION ITÉRATIVE MLP (BLACK-BOX, offline)
#
# Square est une attaque à requêtes : la relancer à CHAQUE batch (comme le
# font AT-FGSM/AT-PGD avec leur gradient exact) serait bien trop coûteux.
# On adopte donc pour le MLP le même schéma "génère → accumule → réentraîne"
# que pour XGBoost (cf. augment_xgboost_iterative) : round k → le MLP_k
# courant génère X_adv (Square) → on l'ajoute au train set → on réentraîne
# un MLP FRAIS dessus (mêmes seeds, même early stopping sur X_val) → MLP_{k+1}.
# ══════════════════════════════════════════════════════════════

def augment_mlp_iterative(mlp_w, X_train, y_train, X_val, y_val, input_size,
                          save_dir, device, attack, eps, n_rounds,
                          epochs, patience, batch_size, lr, bb_max_queries=300):
    fpath = save_dir / f"mlp_at_{attack}.pt"
    if fpath.exists():
        print(f"    {fpath.name} déjà présent → chargement direct")
        model = MLP(input_size=input_size).to(device)
        model.load_state_dict(torch.load(fpath, map_location=device))
        model.eval()
        return MLPWrapper(model, device)

    set_all_seeds(SEED)
    current = mlp_w   # point de départ : MLP baseline (non défendu), déjà entraîné
    X_aug, y_aug = X_train.copy(), y_train.copy()
    mask  = (y_train == 1)
    X_atk = X_train[mask].astype(np.float32)
    y_atk = y_train[mask]

    for r in range(1, n_rounds + 1):
        print(f"\n    ── Round {r}/{n_rounds} ──")
        print(f"    Génération X_adv sur MLP courant ({attack}, eps={eps})...")
        if attack != "square":
            raise ValueError(f"attaque inconnue pour augment_mlp_iterative : {attack}")
        X_adv = square_attack(current, X_atk, y_atk, eps, max_queries=bb_max_queries)

        X_aug = np.concatenate([X_aug, X_adv], axis=0)
        y_aug = np.concatenate([y_aug, y_atk], axis=0)
        print(f"    Dataset cumulé : {len(X_aug)} exemples")

        new_model = MLP(input_size=input_size).to(device)
        n_pos = max(int((y_aug == 1).sum()), 1)
        n_neg = int((y_aug == 0).sum())
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device))
        optimizer = optim.Adam(new_model.parameters(), lr=lr)

        X_t = torch.tensor(X_aug, dtype=torch.float32)
        y_t = torch.tensor(y_aug, dtype=torch.float32).view(-1, 1)
        gen = torch.Generator(); gen.manual_seed(SEED)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=batch_size, shuffle=True, generator=gen)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

        best_f1, no_improve, best_state = -1.0, 0, None
        for epoch in range(epochs):
            new_model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(new_model(xb), yb).backward()
                optimizer.step()

            new_model.eval()
            with torch.no_grad():
                proba = torch.sigmoid(new_model(X_val_t)).cpu().numpy().flatten()
            f1 = f1_score(y_val, (proba >= THRESHOLD).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, no_improve = f1, 0
                best_state = {k: v.cpu().clone() for k, v in new_model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stop epoch {epoch+1} — best val F1 {best_f1:.4f}")
                    break

        new_model.load_state_dict(best_state)
        new_model.eval()
        current = MLPWrapper(new_model, device)
        print(f"    MLP round {r} fitté ✓ (best val F1 {best_f1:.4f})")

    torch.save(current.model.state_dict(), fpath)
    print(f"\n    Sauvegardé : {fpath.name}")
    return current


# ══════════════════════════════════════════════════════════════
# DÉFENSE 3 — AUGMENTATION ADVERSE LogReg
# LogReg n'a pas de boucle PyTorch : on génère X_adv une fois, on refit.
# attack : "fgsm" | "pgd" (white-box, gradient exact) | "square" (black-box,
#          à requêtes — bb_max_queries contrôle le coût).
# ══════════════════════════════════════════════════════════════

C_GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
EVAL_EPS_FOR_SELECTION = 0.1   # eps auquel les attaques sont ÉVALUÉES (evaluate.py)
MIN_F1_RATIO = 0.3             # garde-fou LÂCHE : élimine seulement les modèles dégénérés
                                # (ex. "tout classer attaque") — pas un critère de qualité.


def augment_logreg(logreg_w, X_train, y_train, X_val, y_val, save_dir, attack, eps,
                   bb_max_queries=300):
    """
    Augmentation adverse LogReg + sélection de C par validation.

    BUG identifié (2026-07-15) : avec C=1.0 fixe, le refit sur (train + X_adv
    encore étiqueté "attaque") doit concilier des points quasi-identiques à du
    normal mais étiquetés attaque → ‖w‖_1 explose (×4 sur SWaT). Pour un modèle
    linéaire, la sensibilité L∞ est gouvernée par marge/‖w‖_1, pas la marge
    seule : l'ASR à eps=0.1 augmente au lieu de baisser (défense contre-
    productive, vérifié empiriquement : C=1.0 → ASR 51.7% théorique vs 20.1%
    baseline ; C=0.001 → 10.5%, meilleur que baseline).

    Fix : grid search sur C, choisi pour MAXIMISER le score combiné
    (F1_val − ASR_val(FGSM, eps=0.1)) sur X_val — jamais sur X_test.
    F1 pénalise déjà les solutions dégénérées (ex. "tout classer attaque",
    qui a une précision proche de 0) ; MIN_F1_RATIO n'est qu'un filet de
    sécurité très lâche (30% du F1 baseline), pas un critère de qualité —
    évite qu'un score composite mal calibré ne sélectionne un cas pathologique.
    Sur un petit dataset (BATADAL), le compromis F1/robustesse est plus dur
    qu'un seuil fixe à 90% ne peut absorber : d'où le score combiné plutôt
    qu'un filtre dur + repli sur l'ancien C=1.0 (qui réintroduisait le bug).
    """
    fpath = save_dir / f"logreg_aug_{attack}.pkl"
    if fpath.exists():
        print(f"    {fpath.name} déjà présent → chargement direct")
        return LogRegWrapper(joblib.load(fpath))

    set_all_seeds(SEED)
    mask  = (y_train == 1)
    X_atk = X_train[mask].astype(np.float32)
    y_atk = y_train[mask]
    print(f"    Génération X_adv LogReg ({attack}, eps={eps}) sur {len(X_atk)} attaques...")

    if attack == "fgsm":
        X_adv = fgsm_logreg(logreg_w, X_atk, y_atk, eps=eps)
    elif attack == "pgd":
        X_adv = pgd_logreg(logreg_w, X_atk, y_atk, eps=eps,
                           iters=20, restarts=3, verbose=False)
    elif attack == "square":
        X_adv = square_attack(logreg_w, X_atk, y_atk, eps, max_queries=bb_max_queries)
    else:
        raise ValueError(f"attaque inconnue pour augment_logreg : {attack}")

    X_aug = np.concatenate([X_train, X_adv], axis=0)
    y_aug = np.concatenate([y_train, y_atk], axis=0)
    print(f"    Dataset : {len(X_train)} → {len(X_aug)} exemples")

    # Baseline F1 val (garde-fou) — modèle NON augmenté, pour référence.
    base_proba_val = logreg_w.predict_proba(X_val)
    base_pred_val  = (base_proba_val >= THRESHOLD).astype(int)
    base_f1_val    = f1_score(y_val, base_pred_val, zero_division=0)

    print(f"    Sélection de C par validation (F1 val baseline = {base_f1_val:.4f})...")
    all_candidates = []          # (score, C, f1_val, asr_val, model) — TOUS, même dégénérés
    non_degenerate = []          # sous-ensemble filtré (F1 >= MIN_F1_RATIO * baseline)
    for C in C_GRID:
        cand = LogisticRegression(C=C, max_iter=1000, solver="saga",
                                  class_weight="balanced", random_state=SEED)
        cand.fit(X_aug, y_aug)
        cand_w = LogRegWrapper(cand)

        proba_val = cand_w.predict_proba(X_val)
        pred_val  = (proba_val >= THRESHOLD).astype(int)
        f1_val    = f1_score(y_val, pred_val, zero_division=0)

        tp_mask = (y_val == 1) & (pred_val == 1)
        if tp_mask.sum() == 0:
            asr_val = 1.0   # dégénéré (rien détecté) → pire cas, écarté par le score combiné
        else:
            X_val_atk = X_val[tp_mask].astype(np.float32)
            y_val_atk = y_val[tp_mask]
            X_val_adv = fgsm_logreg(cand_w, X_val_atk, y_val_atk, eps=EVAL_EPS_FOR_SELECTION)
            pred_adv  = (cand_w.predict_proba(X_val_adv) >= THRESHOLD).astype(int)
            asr_val   = float((pred_adv == 0).mean())

        degenerate = f1_val < MIN_F1_RATIO * base_f1_val
        score = f1_val - asr_val   # à maximiser : bonne détection ET peu évadable
        print(f"      C={C:<8g} F1_val={f1_val:.4f}  ASR_val(FGSM,eps={EVAL_EPS_FOR_SELECTION})="
              f"{asr_val*100:5.1f}%  score={score:+.4f}"
              f"{'  [dégénéré, écarté]' if degenerate else ''}")

        entry = (score, C, f1_val, asr_val, cand)
        all_candidates.append(entry)
        if not degenerate:
            non_degenerate.append(entry)

    pool = non_degenerate if non_degenerate else all_candidates
    if not non_degenerate:
        print("    ⚠ Tous les candidats jugés dégénérés — on garde le meilleur score quand même")
    pool.sort(key=lambda t: -t[0])
    _, best_C, best_f1, best_asr, new_lr = pool[0]
    print(f"    → C retenu = {best_C}  (F1_val={best_f1:.4f}, ASR_val={best_asr*100:.1f}%, "
          f"score={pool[0][0]:+.4f})")

    print(f"    → C retenu = {best_C}")
    joblib.dump(new_lr, fpath)
    print(f"    Sauvegardé : {fpath.name}")
    return LogRegWrapper(new_lr)


# ══════════════════════════════════════════════════════════════
# DÉFENSE 4 — AUGMENTATION ITÉRATIVE XGBoost (sans proxy)
# Round k : XGB_k génère X_adv (grad numérique, ou requêtes pour Square)
# → on accumule → on refit. Les X_adv correspondent aux VRAIES failles de
# XGBoost, pas à celles d'un MLP.
# ══════════════════════════════════════════════════════════════

def augment_xgboost_iterative(xgb_w, X_train, y_train, X_val, y_val,
                              save_dir, device, attack, eps, n_rounds,
                              bb_max_queries=300):
    fpath = save_dir / f"xgb_iter_{attack}_r{n_rounds}.json"
    if fpath.exists():
        print(f"    {fpath.name} déjà présent → chargement direct")
        m = XGBClassifier(); m.load_model(str(fpath))
        return XGBoostWrapper(m)

    set_all_seeds(SEED)
    current = xgb_w
    X_aug, y_aug = X_train.copy(), y_train.copy()
    mask  = (y_train == 1)
    X_atk = X_train[mask].astype(np.float32)
    y_atk = y_train[mask]

    for r in range(1, n_rounds + 1):
        print(f"\n    ── Round {r}/{n_rounds} ──")
        print(f"    Génération X_adv sur XGB courant ({attack}, eps={eps})...")
        if attack == "fgsm":
            X_adv = fgsm_xgb(current, X_atk, y_atk, eps=eps)
        elif attack == "pgd":
            X_adv = pgd_xgb(current, X_atk, y_atk, eps=eps,
                            iters=20, restarts=3, verbose=False)
        elif attack == "square":
            X_adv = square_attack(current, X_atk, y_atk, eps, max_queries=bb_max_queries)
        else:
            raise ValueError(f"attaque inconnue pour augment_xgboost_iterative : {attack}")

        X_aug = np.concatenate([X_aug, X_adv], axis=0)
        y_aug = np.concatenate([y_aug, y_atk], axis=0)
        print(f"    Dataset cumulé : {len(X_aug)} exemples")

        n_pos = max(int((y_aug == 1).sum()), 1)
        n_neg = int((y_aug == 0).sum())
        new_xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos, eval_metric="logloss",
            early_stopping_rounds=20, tree_method="hist",
            device=device, random_state=SEED, verbosity=0)
        # Early stopping sur le VRAI val set (clean), pas un split de X_aug.
        new_xgb.fit(X_aug, y_aug, eval_set=[(X_val, y_val)], verbose=False)
        current = XGBoostWrapper(new_xgb)
        print(f"    XGB round {r} fitté ✓")

    current.model.save_model(str(fpath))
    print(f"\n    Sauvegardé : {fpath.name}")
    return current


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run():
    args   = parse_args()
    ds     = args.dataset
    save   = artifacts_dir(ds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps    = args.eps_at

    print(f"\n{'═'*60}")
    print(f"  DÉFENSES — {ds.upper()} | device {device} | eps_at {eps} | seuil {THRESHOLD}")
    print(f"{'═'*60}")

    X_train, y_train, X_val, y_val, mlp_w, logreg_w, xgb_w = load_artifacts(save, device)
    input_size = X_train.shape[1]

    print("\n[1/7] Adversarial Training FGSM (white-box) — MLP")
    adversarial_train_mlp(X_train, y_train, X_val, y_val, input_size, save, device,
                          attack="fgsm", eps=eps, epochs=args.epochs,
                          patience=args.patience, batch_size=args.batch_size,
                          lr=args.lr, mix_ratio=args.mix_ratio)

    print("\n[2/7] Adversarial Training PGD (white-box) — MLP")
    adversarial_train_mlp(X_train, y_train, X_val, y_val, input_size, save, device,
                          attack="pgd", eps=eps, epochs=args.epochs,
                          patience=args.patience, batch_size=args.batch_size,
                          lr=args.lr, mix_ratio=args.mix_ratio)

    print("\n[3/7] Augmentation adverse FGSM (white-box) — LogReg")
    augment_logreg(logreg_w, X_train, y_train, X_val, y_val, save, attack="fgsm", eps=eps)

    print("\n[4/7] Augmentation itérative FGSM (white-box) — XGBoost (self, "
          f"{args.xgb_rounds} rounds)")
    augment_xgboost_iterative(xgb_w, X_train, y_train, X_val, y_val, save, device,
                              attack="fgsm", eps=eps, n_rounds=args.xgb_rounds)

    print(f"\n[5/7] Augmentation itérative Square (black-box) — MLP (self, "
          f"{args.bb_rounds} rounds, {args.bb_max_queries} requêtes/round)")
    augment_mlp_iterative(mlp_w, X_train, y_train, X_val, y_val, input_size, save, device,
                          attack="square", eps=eps, n_rounds=args.bb_rounds,
                          epochs=args.epochs, patience=args.patience,
                          batch_size=args.batch_size, lr=args.lr,
                          bb_max_queries=args.bb_max_queries)

    print("\n[6/7] Augmentation adverse Square (black-box) — LogReg")
    augment_logreg(logreg_w, X_train, y_train, X_val, y_val, save, attack="square", eps=eps,
                   bb_max_queries=args.bb_max_queries)

    print(f"\n[7/7] Augmentation itérative Square (black-box) — XGBoost (self, "
          f"{args.bb_rounds} rounds)")
    augment_xgboost_iterative(xgb_w, X_train, y_train, X_val, y_val, save, device,
                              attack="square", eps=eps, n_rounds=args.bb_rounds,
                              bb_max_queries=args.bb_max_queries)

    print(f"\n{'═'*60}")
    print(f"  DONE — modèles défendus dans {save}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    run()