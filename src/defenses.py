"""
defenses.py — Entraînement des modèles DÉFENDUS
════════════════════════════════════════════════

Produit, pour un dataset donné, quatre modèles durcis :
  - MLP AT-FGSM          → mlp_at_fgsm.pt        (adversarial training, Madry-style)
  - MLP AT-PGD           → mlp_at_pgd.pt
  - LogReg Aug-FGSM      → logreg_aug_fgsm.pkl   (augmentation adverse)
  - XGBoost Aug-FGSM-Iter→ xgb_iter_fgsm_r{R}.json (augmentation itérative auto)

Usage :
    python src/defenses.py --dataset swat
    python src/defenses.py --dataset batadal --eps-at 0.3

Choix de design (thèse) : les ATTAQUES sont menées à eps=0.1, mais les
DÉFENSES sont entraînées à eps=0.3 (--eps-at). Un modèle durci contre des
perturbations plus larges généralise mieux vers les plus petites.

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
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES ARTEFACTS
# ══════════════════════════════════════════════════════════════

def load_artifacts(save_dir, device):
    X_train = np.load(save_dir / "X_train.npy")
    y_train = np.load(save_dir / "y_train.npy")
    X_val   = np.load(save_dir / "X_val.npy")
    y_val   = np.load(save_dir / "y_val.npy")

    logreg_w = LogRegWrapper(joblib.load(save_dir / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(save_dir / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    print(f"✓ Artefacts chargés depuis {save_dir}")
    print(f"  X_train {X_train.shape} | X_val {X_val.shape}")
    return X_train, y_train, X_val, y_val, logreg_w, xgb_w


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
# DÉFENSE 3 — AUGMENTATION ADVERSE LogReg
# LogReg n'a pas de boucle PyTorch : on génère X_adv une fois, on refit.
# ══════════════════════════════════════════════════════════════

def augment_logreg(logreg_w, X_train, y_train, save_dir, attack, eps):
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
    else:
        X_adv = pgd_logreg(logreg_w, X_atk, y_atk, eps=eps,
                           iters=20, restarts=3, verbose=False)

    X_aug = np.concatenate([X_train, X_adv], axis=0)
    y_aug = np.concatenate([y_train, y_atk], axis=0)
    print(f"    Dataset : {len(X_train)} → {len(X_aug)} exemples")

    new_lr = LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                class_weight="balanced", random_state=SEED)
    new_lr.fit(X_aug, y_aug)
    joblib.dump(new_lr, fpath)
    print(f"    Sauvegardé : {fpath.name}")
    return LogRegWrapper(new_lr)


# ══════════════════════════════════════════════════════════════
# DÉFENSE 4 — AUGMENTATION ITÉRATIVE XGBoost (sans proxy)
# Round k : XGB_k génère X_adv (grad numérique) → on accumule → on refit.
# Les X_adv correspondent aux VRAIES failles de XGBoost, pas à celles d'un MLP.
# ══════════════════════════════════════════════════════════════

def augment_xgboost_iterative(xgb_w, X_train, y_train, X_val, y_val,
                              save_dir, device, attack, eps, n_rounds):
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
        else:
            X_adv = pgd_xgb(current, X_atk, y_atk, eps=eps,
                            iters=20, restarts=3, verbose=False)

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

    X_train, y_train, X_val, y_val, logreg_w, xgb_w = load_artifacts(save, device)
    input_size = X_train.shape[1]

    print("\n[1/4] Adversarial Training FGSM — MLP")
    adversarial_train_mlp(X_train, y_train, X_val, y_val, input_size, save, device,
                          attack="fgsm", eps=eps, epochs=args.epochs,
                          patience=args.patience, batch_size=args.batch_size,
                          lr=args.lr, mix_ratio=args.mix_ratio)

    print("\n[2/4] Adversarial Training PGD — MLP")
    adversarial_train_mlp(X_train, y_train, X_val, y_val, input_size, save, device,
                          attack="pgd", eps=eps, epochs=args.epochs,
                          patience=args.patience, batch_size=args.batch_size,
                          lr=args.lr, mix_ratio=args.mix_ratio)

    print("\n[3/4] Augmentation adverse FGSM — LogReg")
    augment_logreg(logreg_w, X_train, y_train, save, attack="fgsm", eps=eps)

    print("\n[4/4] Augmentation itérative FGSM — XGBoost (self, "
          f"{args.xgb_rounds} rounds)")
    augment_xgboost_iterative(xgb_w, X_train, y_train, X_val, y_val, save, device,
                              attack="fgsm", eps=eps, n_rounds=args.xgb_rounds)

    print(f"\n{'═'*60}")
    print(f"  DONE — modèles défendus dans {save}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    run()