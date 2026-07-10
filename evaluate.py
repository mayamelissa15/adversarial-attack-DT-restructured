"""
evaluate.py — Évaluation ADAPTATIVE des modèles défendus (white-box)
═════════════════════════════════════════════════════════════════════

Pour chaque modèle défendu, RÉ-EXÉCUTE les attaques white-box (FGSM/PGD/C&W)
DIRECTEMENT contre lui (attaque adaptative = standard de robustesse), puis
compare l'ASR obtenu à la baseline (modèle non défendu) lue dans les JSON
produits par whitebox_run.py.

C'est l'évaluation des défenses à eps=0.1 (celle qui avait été repoussée).
Coût comparable à un run white-box → à lancer en tmux.

Usage :
    python src/evaluate.py --dataset swat    --eps 0.1 --attack all
    python src/evaluate.py --dataset batadal --eps 0.1 --fast --n_runs 3

Sorties :
    results/<ds>/defense_results.json
    results/<ds>/defense_asr_whitebox.png   (si matplotlib dispo)

Prérequis :
    - defenses.py déjà lancé (modèles défendus dans artifacts/<ds>/)
    - whitebox_run.py déjà lancé au même eps (pour la baseline ; sinon ΔASR = None)
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
import joblib
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from common import THRESHOLD, set_all_seeds, artifacts_dir, results_dir, eval_attack
from models import MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper
from common_whitebox import get_device, eval_sizes, build_per_model_eval
from whitebox_run import run_one_attack, ATTACK_LABEL, ATTACK_FILE


# Matrice des modèles défendus : (modèle de base, nom défense, fichier, type).
DEFENDED_SPEC = [
    ("MLP",     "AT-FGSM",        "mlp_at_fgsm.pt",        "mlp"),
    ("MLP",     "AT-PGD",         "mlp_at_pgd.pt",         "mlp"),
    ("LogReg",  "Aug-FGSM",       "logreg_aug_fgsm.pkl",   "logreg"),
    ("XGBoost", "Aug-FGSM-Iter",  "xgb_iter_fgsm_r3.json", "xgb"),
]


# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Évaluation adaptative des défenses")
    p.add_argument("--dataset", required=True, choices=["swat", "batadal"])
    p.add_argument("--attack",  default="all", choices=["fgsm", "pgd", "cw", "all"])
    p.add_argument("--eps",     type=float, default=0.1)
    p.add_argument("--n_runs",  type=int,   default=3)
    p.add_argument("--fast",    action="store_true")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# CHARGEMENT D'UN MODÈLE DÉFENDU  → (wrapper, is_lr, is_xgb)
# ══════════════════════════════════════════════════════════════

def load_defended(kind, path, input_size, device):
    if kind == "mlp":
        m = MLP(input_size=input_size).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        return MLPWrapper(m, device), False, False
    if kind == "logreg":
        return LogRegWrapper(joblib.load(path)), True, False
    if kind == "xgb":
        m = XGBClassifier(); m.load_model(str(path))
        return XGBoostWrapper(m), False, True
    raise ValueError(kind)


# ══════════════════════════════════════════════════════════════
# BASELINE : ASR médian du modèle NON défendu (depuis whitebox_run JSON)
# ══════════════════════════════════════════════════════════════

def load_baseline_asr(res_dir, ds, eps, attacks):
    """Retourne {(base_model, LABEL): asr_median_%} depuis les JSON white-box."""
    baseline = {}
    for attack in attacks:
        afile = ATTACK_FILE[attack]
        label = ATTACK_LABEL[attack]
        p = res_dir / f"whitebox_{afile}_{ds}_eps{eps}.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        for model_name, atk_dict in data.items():
            if label in atk_dict:
                baseline[(model_name, label)] = atk_dict[label]["evasion_rate_median"]
    return baseline


# ══════════════════════════════════════════════════════════════
# GRAPHE baseline vs défenses
# ══════════════════════════════════════════════════════════════

def plot_defense(summary, baseline, attacks_labels, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib indisponible, graphe ignoré : {e})")
        return

    base_models = ["MLP", "LogReg", "XGBoost"]
    fig, axes = plt.subplots(1, len(base_models),
                             figsize=(5 * len(base_models), 4.5), sharey=True)
    if len(base_models) == 1:
        axes = [axes]

    x = np.arange(len(attacks_labels))
    for ax, bm in zip(axes, base_models):
        # séries : baseline + chaque défense de ce modèle de base
        series = {"baseline": [baseline.get((bm, lb), np.nan) for lb in attacks_labels]}
        for (b, defense), per_attack in summary.items():
            if b != bm:
                continue
            series[defense] = [per_attack.get(lb, {}).get("asr_median", np.nan)
                               for lb in attacks_labels]

        n = len(series)
        w = 0.8 / max(n, 1)
        for i, (name, vals) in enumerate(series.items()):
            ax.bar(x + (i - n / 2 + 0.5) * w, vals, width=w * 0.9, label=name, zorder=3)
        ax.set_title(bm)
        ax.set_xticks(x); ax.set_xticklabels(attacks_labels)
        ax.set_ylim(0, 105)
        ax.set_ylabel("ASR (%)")
        ax.yaxis.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("ASR baseline vs défenses (white-box)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path.name}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    ds     = args.dataset
    save   = artifacts_dir(ds)
    res    = results_dir(ds)
    device = get_device()
    eps    = args.eps

    attacks = ["fgsm", "pgd", "cw"] if args.attack == "all" else [args.attack]
    labels  = [ATTACK_LABEL[a] for a in attacks]
    SEEDS   = list(range(args.n_runs))
    EVAL_ATK, EVAL_NRM = eval_sizes(ds)

    PGD_ITERS    = 50  if args.fast else 200
    PGD_RESTARTS = 3   if args.fast else 10
    CW_ITERS     = 150 if args.fast else 500

    print(f"\n{'═'*60}")
    print(f"  ÉVAL DÉFENSES — {ds.upper()} | eps {eps} | seuil {THRESHOLD}")
    print(f"  Attaques : {', '.join(labels)} | seeds {args.n_runs} | fast {args.fast}")
    print(f"{'═'*60}")

    X_test = np.load(save / "X_test.npy")
    y_test = np.load(save / "y_test.npy")
    input_size = X_test.shape[1]

    baseline = load_baseline_asr(res, ds, eps, attacks)
    if not baseline:
        print("  ⚠ Aucun JSON baseline white-box trouvé — ΔASR sera None.")

    rows = []
    for base_model, defense, fname, kind in DEFENDED_SPEC:
        path = save / fname
        if not path.exists():
            print(f"\n  ✗ {fname} absent — défense ignorée (lance defenses.py).")
            continue

        wrapper, is_lr, is_xgb = load_defended(kind, path, input_size, device)
        print(f"\n  ── {base_model} / {defense} {'─'*30}")

        for seed in SEEDS:
            X_eval, y_eval, X_atk, y_atk, _ = build_per_model_eval(
                X_test, y_test, wrapper, seed, EVAL_ATK, EVAL_NRM, ds)

            for attack in attacks:
                set_all_seeds(seed)
                X_adv = run_one_attack(attack, is_lr, is_xgb, wrapper,
                                       X_atk, y_atk, eps,
                                       PGD_ITERS, PGD_RESTARTS, CW_ITERS)
                r = eval_attack(wrapper, X_eval, y_eval, X_adv,
                                ATTACK_LABEL[attack], base_model, threshold=THRESHOLD)
                rows.append({
                    "base_model": base_model, "defense": defense,
                    "attack": ATTACK_LABEL[attack], "seed": seed,
                    "asr": r["asr"], "f1_adv": r["f1_adv"], "rec_adv": r["rec_adv"],
                })
            print(f"    seed {seed}: " + " | ".join(
                f"{ATTACK_LABEL[a]} {100*np.mean([x['asr'] for x in rows if x['seed']==seed and x['attack']==ATTACK_LABEL[a] and x['defense']==defense]):.1f}%"
                for a in attacks))

    if not rows:
        print("\n  Rien à évaluer. Lance defenses.py d'abord.")
        return

    df = pd.DataFrame(rows)

    # ── Agrégation : médiane ASR par (base_model, defense, attack) ──
    out = {}
    summary = {}
    for (bm, defense), g1 in df.groupby(["base_model", "defense"]):
        out.setdefault(bm, {}).setdefault(defense, {})
        summary[(bm, defense)] = {}
        for label, g2 in g1.groupby("attack"):
            asr_med = round(float(g2["asr"].median()) * 100, 2)
            base    = baseline.get((bm, label))
            entry = {
                "asr_median":    asr_med,
                "asr_std":       round(float(g2["asr"].std(ddof=0)) * 100, 2),
                "f1_adv_median": round(float(g2["f1_adv"].median()), 4),
                "baseline_asr":  base,
                "delta_asr":     (round(asr_med - base, 2) if base is not None else None),
            }
            out[bm][defense][label] = entry
            summary[(bm, defense)][label] = entry

    json_path = res / "defense_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ JSON → {json_path}")

    # ── Résumé console ──
    print(f"\n{'═'*60}\n  RÉSUMÉ — ASR défendu (Δ vs baseline)\n{'═'*60}")
    for bm in ["MLP", "LogReg", "XGBoost"]:
        for defense, per_attack in out.get(bm, {}).items():
            for label, e in per_attack.items():
                d = e["delta_asr"]
                d_str = f"Δ {d:+.1f}" if d is not None else "Δ  n/a"
                print(f"  {bm:8s}/{defense:14s} {label:5s}  "
                      f"ASR {e['asr_median']:5.1f}%  {d_str}")

    plot_defense(summary, baseline, labels, res / "defense_asr_whitebox.png")


if __name__ == "__main__":
    main()
