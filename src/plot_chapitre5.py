"""
plot_chapitre5.py — Figures du Chapitre 5 (résultats d'attaques) du mémoire.

Lit DIRECTEMENT les JSON agrégés (déjà multi-seed : médiane + écart-type)
dans results/<dataset>/ :
    whitebox_fgsm_<ds>_eps<eps>.json
    whitebox_pgd_<ds>_eps<eps>.json
    whitebox_cw_<ds>_eps<eps>.json
    blackbox_score_<ds>_eps<eps>.json
    blackbox_decision_<ds>_eps<eps>.json
    blackbox_transfer_<ds>_eps<eps>.json

Sortie : figures/chapitre5/*.png (vectoriel non demandé ici — PNG haute
résolution, dpi=300, pas de titre dans la figure — le \\caption est géré
côté LaTeX).

Usage :
    python src/plot_chapitre5.py
    python src/plot_chapitre5.py --eps 0.1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR     = ROOT / "figures" / "chapitre5"

# ══════════════════════════════════════════════════════════════
# STYLE — cohérent sur tout le mémoire (voir consignes Chapitre 5)
# ══════════════════════════════════════════════════════════════

MODEL_COLOR = {
    "MLP":     "#0072B2",
    "LogReg":  "#E69F00",
    "XGBoost": "#009E73",
}
MODEL_ORDER = ["MLP", "LogReg", "XGBoost"]
GRID_COLOR  = "#CCCCCC"
BAR_ALPHA   = 0.9

plt.rcParams.update({
    "font.size":        9,
    "axes.labelsize":   9.5,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "font.family":      "sans-serif",
    "savefig.dpi":      300,
    "pdf.fonttype":     42,   # texte éditable si jamais export PDF
})


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, ls="--", linewidth=0.7, alpha=1.0, color=GRID_COLOR, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


# ══════════════════════════════════════════════════════════════
# CHARGEMENT JSON
# ══════════════════════════════════════════════════════════════

def _load(res_dir, name):
    path = res_dir / name
    if not path.exists():
        print(f"  (absent, ignoré) {name}")
        return None
    with open(path) as f:
        return json.load(f)


def load_whitebox(ds, eps):
    """→ dict[model][attack] = (median, std) pour FGSM / PGD / C&W."""
    res_dir = RESULTS_DIR / ds
    out = {}
    for fname, attack_key in [
        (f"whitebox_fgsm_{ds}_eps{eps}.json", "FGSM"),
        (f"whitebox_pgd_{ds}_eps{eps}.json",  "PGD"),
        (f"whitebox_cw_{ds}_eps{eps}.json",   "C&W"),
    ]:
        data = _load(res_dir, fname)
        if data is None:
            continue
        for model, attacks in data.items():
            m = attacks.get(attack_key)
            if m is None:
                continue
            out.setdefault(model, {})[attack_key] = (
                m["evasion_rate_median"], m.get("evasion_rate_std", 0.0)
            )
    return out


def load_blackbox_family(ds, eps, family):
    """family in {'score', 'decision', 'transfer'} → dict[model][attack] = (median, std)."""
    res_dir = RESULTS_DIR / ds
    data = _load(res_dir, f"blackbox_{family}_{ds}_eps{eps}.json")
    if data is None:
        return {}
    out = {}
    for model, attacks in data.items():
        for attack, m in attacks.items():
            out.setdefault(model, {})[attack] = (
                m["evasion_rate_median"], m.get("evasion_rate_std", 0.0)
            )
    return out


# ══════════════════════════════════════════════════════════════
# FIGURE — barres groupées par attaque, une couleur par modèle
# (utilisée pour whitebox / score-based / decision-based)
# ══════════════════════════════════════════════════════════════

def bar_grouped_by_attack(data, attack_order, out_path, ylabel="ASR médiane (%)"):
    models  = [m for m in MODEL_ORDER if m in data]
    attacks = [a for a in attack_order if any(a in data.get(m, {}) for m in models)]
    if not models or not attacks:
        print(f"  (rien à tracer) {out_path.name}")
        return

    n_models = len(models)
    x = np.arange(len(attacks))
    w = 0.78 / n_models

    fig, ax = plt.subplots(figsize=(max(4.2, 1.5 * len(attacks) * n_models * 0.55 + 1.5), 3.6))
    _style_axis(ax)

    for i, model in enumerate(models):
        vals, errs = [], []
        for a in attacks:
            med, std = data.get(model, {}).get(a, (np.nan, 0.0))
            vals.append(med)
            errs.append(std)
        xpos = x + (i - n_models / 2 + 0.5) * w
        ax.bar(xpos, vals, width=w * 0.9, color=MODEL_COLOR[model], label=model,
               alpha=BAR_ALPHA, edgecolor="black", linewidth=0.5, zorder=3)
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel(ylabel)
    ymax = np.nanmax([data.get(m, {}).get(a, (0, 0))[0] + data.get(m, {}).get(a, (0, 0))[1]
                      for m in models for a in attacks] + [10])
    ax.set_ylim(0, min(ymax * 1.18, 108))
    ax.legend(frameon=False, loc="upper right", ncol=n_models, handlelength=1.2,
              columnspacing=1.0, handletextpad=0.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ══════════════════════════════════════════════════════════════
# FIGURE — transfert : groupée par MODÈLE VICTIME, une barre par
# substitut × attaque (Sub1/2/3 × MI-FGSM/VMI-FGSM), + Ensemble-MI
# séparé visuellement en fin de groupe.
# Couleur = modèle victime (cohérence de palette) ; hachure = variante
# substitut/attaque (seule dimension qui varie DANS un groupe).
# ══════════════════════════════════════════════════════════════

SUB_ORDER = [
    ("MI-FGSM_Sub1-MLP",       "S1 · MI-FGSM",  ""),
    ("VMI-FGSM_Sub1-MLP",      "S1 · VMI-FGSM", ".."),
    ("MI-FGSM_Sub2-SmallMLP",  "S2 · MI-FGSM",  "//"),
    ("VMI-FGSM_Sub2-SmallMLP", "S2 · VMI-FGSM", "//.."),
    ("MI-FGSM_Sub3-DeepMLP",   "S3 · MI-FGSM",  "xx"),
    ("VMI-FGSM_Sub3-DeepMLP",  "S3 · VMI-FGSM", "xx.."),
]
ENSEMBLE_KEY   = "Ensemble-MI"
ENSEMBLE_LABEL = "Ensemble-MI"
ENSEMBLE_HATCH = "oo"


def bar_transfer(data, out_path, ylabel="ASR médiane (%)"):
    models = [m for m in MODEL_ORDER if m in data]
    if not models:
        print(f"  (rien à tracer) {out_path.name}")
        return

    n_slots   = len(SUB_ORDER) + 1        # 6 substituts + Ensemble-MI
    bar_w     = 0.115
    extra_gap = 0.05                       # séparation visuelle avant Ensemble-MI
    # positions relatives (centrées ensuite autour de 0)
    rel_x = [i * bar_w for i in range(len(SUB_ORDER))]
    rel_x.append(rel_x[-1] + bar_w + extra_gap)
    rel_x = np.array(rel_x)
    rel_x -= rel_x.mean()

    fig, ax = plt.subplots(figsize=(max(6.5, 2.6 * len(models)), 3.9))
    _style_axis(ax)

    group_centers = np.arange(len(models))
    ymax = 10.0
    for gi, model in enumerate(models):
        color = MODEL_COLOR[model]
        for (key, _label, hatch), xr in zip(SUB_ORDER, rel_x[:-1]):
            med, std = data.get(model, {}).get(key, (np.nan, 0.0))
            xpos = group_centers[gi] + xr
            ax.bar(xpos, med, width=bar_w * 0.92, color=color, alpha=BAR_ALPHA,
                   edgecolor="black", linewidth=0.5, hatch=hatch, zorder=3)
            ax.errorbar(xpos, med, yerr=std, fmt="none", ecolor="black",
                        elinewidth=0.6, capsize=1.8, capthick=0.6, zorder=4)
            if not np.isnan(med):
                ymax = max(ymax, med + std)

        med, std = data.get(model, {}).get(ENSEMBLE_KEY, (np.nan, 0.0))
        xpos = group_centers[gi] + rel_x[-1]
        ax.bar(xpos, med, width=bar_w * 0.92, color=color, alpha=BAR_ALPHA,
               edgecolor="black", linewidth=1.0, hatch=ENSEMBLE_HATCH, zorder=3)
        ax.errorbar(xpos, med, yerr=std, fmt="none", ecolor="black",
                    elinewidth=0.6, capsize=1.8, capthick=0.6, zorder=4)
        if not np.isnan(med):
            ymax = max(ymax, med + std)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, min(ymax * 1.28, 112))

    # légende (dimension hachure = substitut/attaque), swatches neutres en gris
    handles = [Patch(facecolor="white", edgecolor="black", hatch=h, label=lbl)
              for (_k, lbl, h) in SUB_ORDER]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch=ENSEMBLE_HATCH,
                         label=ENSEMBLE_LABEL))
    ax.legend(handles=handles, frameon=False, loc="upper center",
             bbox_to_anchor=(0.5, 1.24), ncol=4, fontsize=8, handlelength=1.6,
             columnspacing=1.0, handletextpad=0.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_dataset(ds, eps):
    print(f"\n{'═'*60}\n  CHAPITRE 5 — {ds.upper()}\n{'═'*60}")

    wb = load_whitebox(ds, eps)
    bar_grouped_by_attack(wb, ["FGSM", "PGD", "C&W"],
                          OUT_DIR / f"fig_whitebox_{ds}.png")

    sc = load_blackbox_family(ds, eps, "score")
    bar_grouped_by_attack(sc, ["Square", "NES"],
                          OUT_DIR / f"fig_score_{ds}.png")

    dc = load_blackbox_family(ds, eps, "decision")
    bar_grouped_by_attack(dc, ["HSJA", "RayS"],
                          OUT_DIR / f"fig_decision_{ds}.png")

    tr = load_blackbox_family(ds, eps, "transfer")
    bar_transfer(tr, OUT_DIR / f"fig_transfer_{ds}.png")


def parse_args():
    p = argparse.ArgumentParser(description="Génère les figures du Chapitre 5 depuis results/*.json")
    p.add_argument("--dataset", default="both", choices=["swat", "batadal", "both"])
    p.add_argument("--eps", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    datasets = ["batadal", "swat"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, args.eps)
    print(f"\n→ figures dans {OUT_DIR}")


if __name__ == "__main__":
    main()
