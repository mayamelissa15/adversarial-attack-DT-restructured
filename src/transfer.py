"""
transfer.py
═══════════
Attaques black-box par TRANSFERT : on entraîne des substituts (surrogates),
on génère des exemples adverses dessus (gradient du substitut), puis on les
transfère vers la victime (qu'on ne différencie jamais).

Contenu :
  - train_substitute()  : entraîne un MLP substitut (avec bruit d'entrée)
  - mi_fgsm()           : Momentum Iterative FGSM
  - vmi_fgsm()          : Variance-tuned MI-FGSM
  - ensemble_mi_fgsm()  : MI-FGSM sur un ensemble de substituts
  - ensemble_vmi_fgsm() : VMI-FGSM ensemble (dispo, non utilisé par défaut)
  - eval_transfer()     : mesure l'ASR de X_adv (substitut) sur la victime

Refactor vs ancienne version :
  - Seuil = common.THRESHOLD (0.5) au lieu de 0.45.
  - train_substitute early stopping sur le VRAI X_val (avant : X_train[:5000]).
  - Plus de bloc __main__ ni de save_all_adv (orchestration → blackbox_run.py).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score

from common import THRESHOLD
from models import MLPWrapper  # arch classes passées en argument

ITERS = 40   # itérations par défaut des attaques itératives


# ══════════════════════════════════════════════════════════════
# ENTRAÎNEMENT D'UN SUBSTITUT
# ══════════════════════════════════════════════════════════════

def train_substitute(arch_class, X_train, y_train, X_val, y_val,
                     device, name="Sub", noise_std=0.02):
    """
    Entraîne un MLP substitut. Le bruit gaussien sur les entrées (noise_std)
    diversifie les substituts d'un seed à l'autre → meilleure transférabilité.
    Early stopping sur le VRAI validation set.
    """
    model = arch_class(input_size=X_train.shape[1]).to(device)

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = int((y_train == 0).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=2048, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_f1, patience, no_improve, best_state = -1.0, 5, 0, None
    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
        f1 = f1_score(y_val, (proba >= THRESHOLD).astype(int), zero_division=0)

        if f1 > best_f1:
            best_f1, no_improve = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"    {name} — best val F1 : {best_f1:.4f}")
    return MLPWrapper(model, device)


# ══════════════════════════════════════════════════════════════
# MI-FGSM
# ══════════════════════════════════════════════════════════════

def mi_fgsm(sub_wrapper, X_atk, y_atk, eps, iters=ITERS, mu=1.0):
    alpha  = 2 * eps / iters
    device = sub_wrapper.device
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_orig)

    sub_wrapper.model.eval()
    for _ in range(iters):
        x_adv  = x_adv.detach().requires_grad_(True)
        logits = sub_wrapper.model(x_adv)
        nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction='sum').backward()

        grad      = x_adv.grad.data
        grad_norm = grad / (grad.abs().sum(dim=1, keepdim=True) + 1e-12)  # normalisation L1
        g         = mu * g + grad_norm
        x_adv     = x_adv.detach() + alpha * g.sign()
        x_adv     = torch.clamp(x_adv, x_orig - eps, x_orig + eps)
    return x_adv.cpu().numpy()


def ensemble_mi_fgsm(sub_wrappers, X_atk, y_atk, eps,
                     iters=ITERS, mu=1.0, weights=None):
    if weights is None:
        weights = [1.0 / len(sub_wrappers)] * len(sub_wrappers)
    alpha  = 2 * eps / iters
    device = sub_wrappers[0].device
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_orig)

    for sub in sub_wrappers:
        sub.model.eval()

    for _ in range(iters):
        x_adv_d       = x_adv.detach()
        grad_ensemble = torch.zeros_like(x_orig)
        for w, sub in zip(weights, sub_wrappers):
            x_inp  = x_adv_d.requires_grad_(True)
            logits = sub.model(x_inp)
            nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction='sum').backward()
            grad_cur  = x_inp.grad.data.clone()
            grad_norm = grad_cur / (grad_cur.abs().sum(dim=1, keepdim=True) + 1e-12)
            grad_ensemble += w * grad_norm

        g     = mu * g + grad_ensemble
        x_adv = x_adv_d + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()
    return x_adv.cpu().numpy()


# ══════════════════════════════════════════════════════════════
# VMI-FGSM  (variance-tuned)
# ══════════════════════════════════════════════════════════════

def vmi_fgsm(sub_wrapper, X_atk, y_atk, eps,
             iters=ITERS, mu=1.0, beta=0.3, n_neighbors=10):
    alpha  = 2 * eps / iters
    device = sub_wrapper.device
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_orig)

    sub_wrapper.model.eval()
    for _ in range(iters):
        x_adv_d = x_adv.detach()

        # Gradient au point courant
        x_inp  = x_adv_d.requires_grad_(True)
        logits = sub_wrapper.model(x_inp)
        nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction='sum').backward()
        grad_cur = x_inp.grad.data.clone()

        # Moyenne des gradients dans un voisinage aléatoire
        grad_neigh = torch.zeros_like(grad_cur)
        for _ in range(n_neighbors):
            noise    = torch.empty_like(x_adv_d).uniform_(-beta * eps, beta * eps)
            x_n      = (x_adv_d + noise).detach().requires_grad_(True)
            logits_n = sub_wrapper.model(x_n)
            nn.functional.binary_cross_entropy_with_logits(logits_n, y_t, reduction='sum').backward()
            grad_neigh += x_n.grad.data
        grad_neigh /= n_neighbors

        # Correction de variance + normalisation L1 du gradient final
        grad_used      = grad_cur + beta * (grad_cur - grad_neigh)
        grad_used_norm = grad_used / (grad_used.abs().sum(dim=1, keepdim=True) + 1e-12)

        g     = mu * g + grad_used_norm
        x_adv = x_adv_d + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()
    return x_adv.cpu().numpy()


def ensemble_vmi_fgsm(sub_wrappers, X_atk, y_atk, eps,
                      iters=ITERS, mu=1.0, beta=0.3, n_neighbors=10, weights=None):
    if weights is None:
        weights = [1.0 / len(sub_wrappers)] * len(sub_wrappers)
    alpha  = 2 * eps / iters
    device = sub_wrappers[0].device
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_orig)

    for sub in sub_wrappers:
        sub.model.eval()

    for _ in range(iters):
        x_adv_d       = x_adv.detach()
        grad_ensemble = torch.zeros_like(x_orig)
        for w, sub in zip(weights, sub_wrappers):
            x_inp  = x_adv_d.requires_grad_(True)
            logits = sub.model(x_inp)
            nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction='sum').backward()
            grad_cur = x_inp.grad.data.clone()

            grad_neigh = torch.zeros_like(grad_cur)
            for _ in range(n_neighbors):
                noise    = torch.empty_like(x_adv_d).uniform_(-beta * eps, beta * eps)
                x_n      = (x_adv_d + noise).detach().requires_grad_(True)
                logits_n = sub.model(x_n)
                nn.functional.binary_cross_entropy_with_logits(logits_n, y_t, reduction='sum').backward()
                grad_neigh += x_n.grad.data
            grad_neigh /= n_neighbors

            grad_used      = grad_cur + beta * (grad_cur - grad_neigh)
            grad_used_norm = grad_used / (grad_used.abs().sum(dim=1, keepdim=True) + 1e-12)
            grad_ensemble += w * grad_used_norm

        g     = mu * g + grad_ensemble
        x_adv = x_adv_d + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()
    return x_adv.cpu().numpy()


# ══════════════════════════════════════════════════════════════
# ÉVALUATION DU TRANSFERT
# ══════════════════════════════════════════════════════════════

def eval_transfer(X_eval, y_eval, X_adv_atk,
                  victim_wrapper, sub_name, victim_name, attack_name):
    """
    ASR = fraction des attaques (perturbées sur le substitut) que la VICTIME
    classe désormais comme normales. Le seuil est common.THRESHOLD.
    """
    mask             = (y_eval == 1)
    X_adv_full       = X_eval.copy()
    X_adv_full[mask] = X_adv_atk

    pr_clean = (victim_wrapper.predict_proba(X_eval)     >= THRESHOLD).astype(int)
    pr_adv   = (victim_wrapper.predict_proba(X_adv_full) >= THRESHOLD).astype(int)

    f1_clean  = f1_score(y_eval, pr_clean, zero_division=0)
    f1_adv    = f1_score(y_eval, pr_adv,   zero_division=0)
    rec_clean = recall_score(y_eval, pr_clean, zero_division=0)
    rec_adv   = recall_score(y_eval, pr_adv,   zero_division=0)

    pr_att = (victim_wrapper.predict_proba(X_adv_atk) >= THRESHOLD).astype(int)
    asr    = float(np.mean(pr_att == 0))
    linf   = float(np.abs(X_adv_atk - X_eval[mask]).max())

    print(f"    [{attack_name}] {sub_name} → {victim_name:8s} | "
          f"ASR {asr:.1%} | F1 {f1_clean:.3f}→{f1_adv:.3f} | L∞ {linf:.4f}")

    return {
        "attack": attack_name, "substitute": sub_name, "victim": victim_name,
        "asr": round(asr, 4),
        "f1_clean": round(f1_clean, 4), "f1_adv": round(f1_adv, 4),
        "rec_clean": round(rec_clean, 4), "rec_adv": round(rec_adv, 4),
        "delta_f1": round(f1_adv - f1_clean, 4), "linf": round(linf, 4),
    }