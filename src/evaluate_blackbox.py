"""
evaluate_blackbox.py — Évaluation ADAPTATIVE des modèles défendus (black-box)
══════════════════════════════════════════════════════════════════════════

Pendant de evaluate.py, mais pour les attaques BLACK-BOX. Pour chaque modèle
défendu (durci white-box OU black-box, cf. DEFENDED_SPEC), RÉ-EXÉCUTE les
attaques DIRECTEMENT contre lui (attaque adaptative) :

  - Score-based    : Square, NES
  - Decision-based : HSJA, RayS       (sous-échantillonné, coûteux)
  - Transfer       : MI-FGSM, VMI-FGSM, Ensemble-MI (substituts entraînés sur
                     données CLEAN — jamais sur le modèle défendu — un seul
                     jeu de substituts par seed, réutilisé pour tous les
                     modèles défendus de ce seed)

Compare l'ASR obtenu à la baseline (modèle NON défendu) lue dans les JSON
produits par blackbox_run.py --victims baseline.

Usage :
    python src/evaluate_blackbox.py --dataset swat    --eps 0.1 --family all
    python src/evaluate_blackbox.py --dataset batadal --eps 0.1 --fast --n_runs 2 --family score

Sorties :
    results/<ds>/defense_results_blackbox.json
    results/<ds>/defense_asr_blackbox.png   (si matplotlib dispo)

Prérequis :
    - defenses.py déjà lancé (modèles défendus dans artifacts/<ds>/, y compris
      les variantes black-box mlp_at_square.pt / logreg_aug_square.pkl /
      xgb_iter_square_r3.json)
    - blackbox_run.py --victims baseline déjà lancé au même eps (pour la
      baseline ; sinon ΔASR = None)
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import (THRESHOLD, set_all_seeds, artifacts_dir, results_dir, eval_attack,
                    subsample_for_substitute)
from common_whitebox import get_device, eval_sizes, build_per_model_eval
from models import MLP, SmallMLP, DeepMLP
from blackbox import square_attack, nes_attack, hsja, rays
from transfer import mi_fgsm, vmi_fgsm, ensemble_mi_fgsm, train_substitute, eval_transfer
from evaluate import load_defended


# Matrice des modèles défendus évalués — white-box ET black-box hardened,
# pour voir si un durcissement white-box tient aussi contre le black-box
# (et réciproquement).
DEFENDED_SPEC = [
    ("MLP",     "AT-FGSM",         "mlp_at_fgsm.pt",          "mlp"),
    ("MLP",     "AT-PGD",          "mlp_at_pgd.pt",           "mlp"),
    ("MLP",     "AT-Square",       "mlp_at_square.pt",        "mlp"),
    ("LogReg",  "Aug-FGSM",        "logreg_aug_fgsm.pkl",     "logreg"),
    ("LogReg",  "Aug-Square",      "logreg_aug_square.pkl",   "logreg"),
    ("XGBoost", "Aug-FGSM-Iter",   "xgb_iter_fgsm_r3.json",   "xgb"),
    ("XGBoost", "Aug-Square-Iter", "xgb_iter_square_r3.json", "xgb"),
]

SCORE_ATTACKS = [("Square", square_attack), ("NES", nes_attack)]


# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Évaluation adaptative des défenses (black-box)")
    p.add_argument("--dataset", required=True, choices=["swat", "batadal"])
    p.add_argument("--eps",     type=float, default=0.1)
    p.add_argument("--n_runs",  type=int,   default=3)
    p.add_argument("--family",  default="all", choices=["score", "transfer", "decision", "all"])
    p.add_argument("--fast",    action="store_true")
    p.add_argument("--max_db",  type=int, default=None,
                   help="Nb max d'attaques pour decision-based (défaut : 100 en --fast, 300 sinon)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# BASELINE : ASR médian du modèle NON défendu (depuis blackbox_run JSON)
# ══════════════════════════════════════════════════════════════

def load_baseline_asr(res_dir, ds, eps):
    """Retourne {(model, LABEL): asr_median_%} depuis les JSON black-box baseline."""
    baseline = {}
    for family in ("score", "transfer", "decision"):
        p = res_dir / f"blackbox_{family}_{ds}_eps{eps}.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        for model_name, atk_dict in data.items():
            for label, m in atk_dict.items():
                baseline[(model_name, label)] = m["evasion_rate_median"]
    return baseline


# ══════════════════════════════════════════════════════════════
# GRAPHE baseline vs défenses
# ══════════════════════════════════════════════════════════════

def plot_defense(summary, baseline, attack_labels, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib indisponible, graphe ignoré : {e})")
        return

    base_models = ["MLP", "LogReg", "XGBoost"]
    fig, axes = plt.subplots(1, len(base_models),
                             figsize=(6 * len(base_models), 4.5), sharey=True)
    if len(base_models) == 1:
        axes = [axes]

    x = np.arange(len(attack_labels))
    for ax, bm in zip(axes, base_models):
        series = {"baseline": [baseline.get((bm, lb), np.nan) for lb in attack_labels]}
        for (b, defense), per_attack in summary.items():
            if b != bm:
                continue
            series[defense] = [per_attack.get(lb, {}).get("asr_median", np.nan)
                               for lb in attack_labels]

        n = len(series)
        w = 0.8 / max(n, 1)
        for i, (name, vals) in enumerate(series.items()):
            ax.bar(x + (i - n / 2 + 0.5) * w, vals, width=w * 0.9, label=name, zorder=3)
        ax.set_title(bm)
        ax.set_xticks(x); ax.set_xticklabels(attack_labels, rotation=30, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("ASR (%)")
        ax.yaxis.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=7)

    fig.suptitle("ASR baseline vs défenses (black-box)", fontweight="bold")
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

    SEEDS = list(range(args.n_runs))
    EVAL_ATK, EVAL_NRM = eval_sizes(ds)

    RUN_SCORE    = args.family in ("all", "score")
    RUN_TRANSFER = args.family in ("all", "transfer")
    RUN_DECISION = args.family in ("all", "decision")

    MAX_DB      = args.max_db or (100 if args.fast else 300)
    HSJA_ITERS  = 10  if args.fast else 40
    HSJA_NEST   = 30  if args.fast else 100
    RAYS_ITERS  = 15  if args.fast else 60
    RAYS_SEARCH = 10  if args.fast else 15

    print(f"\n{'═'*60}")
    print(f"  ÉVAL DÉFENSES BLACK-BOX — {ds.upper()} | eps {eps} | seuil {THRESHOLD}")
    print(f"  Familles : score={RUN_SCORE} transfer={RUN_TRANSFER} decision={RUN_DECISION}")
    print(f"  seeds {args.n_runs} | fast {args.fast}")
    print(f"{'═'*60}")

    X_test  = np.load(save / "X_test.npy")
    y_test  = np.load(save / "y_test.npy")
    X_train = np.load(save / "X_train.npy")
    y_train = np.load(save / "y_train.npy")
    X_val   = np.load(save / "X_val.npy")
    y_val   = np.load(save / "y_val.npy")
    X_train, y_train = subsample_for_substitute(X_train, y_train)
    input_size = X_test.shape[1]

    baseline = load_baseline_asr(res, ds, eps)
    if not baseline:
        print("  ⚠ Aucun JSON baseline black-box trouvé "
              "(lance blackbox_run.py --victims baseline) — ΔASR sera None.")

    # ── Préchargement des modèles défendus disponibles ──
    defended = []
    for base_model, defense, fname, kind in DEFENDED_SPEC:
        path = save / fname
        if not path.exists():
            print(f"\n  ✗ {fname} absent — défense ignorée (lance defenses.py).")
            continue
        wrapper, _, _ = load_defended(kind, path, input_size, device)
        defended.append((base_model, defense, wrapper))

    if not defended:
        print("\n  Rien à évaluer. Lance defenses.py d'abord.")
        return

    rows = []
    attack_labels_seen = []

    def record(base_model, defense, label, seed, r):
        rows.append({
            "base_model": base_model, "defense": defense, "attack": label,
            "seed": seed, "asr": r["asr"], "f1_adv": r["f1_adv"], "rec_adv": r["rec_adv"],
        })
        if label not in attack_labels_seen:
            attack_labels_seen.append(label)

    for seed in SEEDS:
        print(f"\n{'═'*60}\n  SEED {seed+1}/{args.n_runs}\n{'═'*60}")

        # Substituts entraînés UNE FOIS par seed (clean), réutilisés pour
        # toutes les défenses — le transfert ne dépend pas de la victime.
        subs = None
        if RUN_TRANSFER:
            set_all_seeds(seed)
            sub1 = train_substitute(MLP,      X_train, y_train, X_val, y_val, device, f"Sub1-seed{seed}")
            sub2 = train_substitute(SmallMLP, X_train, y_train, X_val, y_val, device, f"Sub2-seed{seed}")
            sub3 = train_substitute(DeepMLP,  X_train, y_train, X_val, y_val, device, f"Sub3-seed{seed}")
            subs = [("Sub1-MLP", sub1), ("Sub2-SmallMLP", sub2), ("Sub3-DeepMLP", sub3)]

        for base_model, defense, wrapper in defended:
            print(f"\n  ── {base_model} / {defense} {'─'*30}")
            X_eval, y_eval, X_atk, y_atk, idx_ev = build_per_model_eval(
                X_test, y_test, wrapper, seed, EVAL_ATK, EVAL_NRM, ds)

            # ─────────────── SCORE-BASED ───────────────
            if RUN_SCORE:
                for label, fn in SCORE_ATTACKS:
                    set_all_seeds(seed)
                    X_adv = fn(wrapper, X_atk, y_atk, eps)
                    r = eval_attack(wrapper, X_eval, y_eval, X_adv, label, base_model,
                                    threshold=THRESHOLD)
                    record(base_model, defense, label, seed, r)
                    print(f"    [Score] {label}: ASR {r['asr']*100:.1f}%")

            # ─────────────── DECISION-BASED (sous-échantillonné) ───────────────
            if RUN_DECISION:
                rng = np.random.default_rng(seed)
                n_db = min(MAX_DB, len(X_atk))
                sub_idx = rng.choice(len(X_atk), n_db, replace=False)
                X_atk_db, y_atk_db = X_atk[sub_idx], y_atk[sub_idx]
                mask_nrm = (y_eval == 0)
                X_db = np.concatenate([X_eval[mask_nrm], X_atk_db], axis=0)
                y_db = np.concatenate(
                    [y_eval[mask_nrm], np.ones(len(X_atk_db), dtype=y_eval.dtype)], axis=0)

                for label, fn, kw in [
                    ("HSJA", hsja, {"iters": HSJA_ITERS, "n_est": HSJA_NEST}),
                    ("RayS", rays, {"iters": RAYS_ITERS, "search_steps": RAYS_SEARCH}),
                ]:
                    print(f"    [Decision] {label} sur {len(X_atk_db)} attaques...")
                    set_all_seeds(seed)
                    X_adv_db = fn(wrapper, X_atk_db, y_atk_db, eps, **kw)
                    r = eval_attack(wrapper, X_db, y_db, X_adv_db, label, base_model,
                                    threshold=THRESHOLD)
                    record(base_model, defense, label, seed, r)
                    print(f"      ASR {r['asr']*100:.1f}%")

            # ─────────────── TRANSFER ───────────────
            if RUN_TRANSFER:
                for sub_name, sub_w in subs:
                    set_all_seeds(seed)
                    X_mi  = mi_fgsm(sub_w,  X_atk, y_atk, eps=eps)
                    X_vmi = vmi_fgsm(sub_w, X_atk, y_atk, eps=eps)
                    for label, X_adv in [("MI-FGSM", X_mi), ("VMI-FGSM", X_vmi)]:
                        rt = eval_transfer(X_eval, y_eval, X_adv, wrapper, sub_name,
                                           base_model, label)
                        record(base_model, defense, f"{label}_{sub_name}", seed, rt)

                set_all_seeds(seed)
                sub_ws = [s for _, s in subs]
                X_ens = ensemble_mi_fgsm(sub_ws, X_atk, y_atk, eps=eps,
                                         weights=[1/3, 1/3, 1/3])
                rt = eval_transfer(X_eval, y_eval, X_ens, wrapper, "Ensemble(S1+S2+S3)",
                                   base_model, "Ensemble-MI")
                record(base_model, defense, "Ensemble-MI", seed, rt)

    if not rows:
        print("\n  Rien à évaluer.")
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

    json_path = res / "defense_results_blackbox.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ JSON → {json_path}")

    # ── Résumé console ──
    print(f"\n{'═'*60}\n  RÉSUMÉ — ASR défendu black-box (Δ vs baseline)\n{'═'*60}")
    for bm in ["MLP", "LogReg", "XGBoost"]:
        for defense, per_attack in out.get(bm, {}).items():
            for label, e in per_attack.items():
                d = e["delta_asr"]
                d_str = f"Δ {d:+.1f}" if d is not None else "Δ  n/a"
                print(f"  {bm:8s}/{defense:16s} {label:16s}  ASR {e['asr_median']:5.1f}%  {d_str}")

    plot_defense(summary, baseline, attack_labels_seen, res / "defense_asr_blackbox.png")


if __name__ == "__main__":
    main()
