"""
generate_figures_chapitre7.py — Figures du Chapitre 7 (résultats de
défense) du mémoire, une figure par défense.

Lit DIRECTEMENT les JSON agrégés (déjà multi-seed : médiane + écart-type)
dans results/<dataset>/ :
    defense_results.json            (réévaluation white-box, evaluate.py)
    defense_results_blackbox.json   (réévaluation black-box, evaluate_blackbox.py)

Ne recalcule et ne relance rien : lecture seule de results/. Si un fichier
ou une clé attendue est absente, un warning explicite est loggé et
l'attaque concernée est omise de la figure plutôt que d'afficher un zéro
trompeur.

Sortie : figures/chapitre7/fig_{defense_snake_case}.png (6 figures)

Usage :
    python src/generate_figures_chapitre7.py
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR     = ROOT / "figures" / "chapitre7"
EPS         = 0.1
DATASETS    = [("swat", "SWaT"), ("batadal", "BATADAL")]

# ══════════════════════════════════════════════════════════════
# PALETTE — identique au Chapitre 6, deux couleurs uniquement
# ══════════════════════════════════════════════════════════════

COLOR_BASELINE = "#D9D9D9"  # gris pastel — modèle non défendu
COLOR_DEFENDED = "#B5EAD7"  # vert pastel — modèle défendu

EDGE_COLOR = "#4A4A4A"
EDGE_WIDTH = 0.6

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi":      300,
})


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls=":", linewidth=0.8, alpha=0.3, color="black", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════
# DÉFENSES — modèle associé, et attaques testées par défense
# ══════════════════════════════════════════════════════════════

WB_ATTACKS = ["FGSM", "PGD", "C&W"]
BB_ATTACKS = ["Square", "NES", "HSJA", "RayS"]
ENSEMBLE_ATTACK = "Ensemble-MI"

DEFENSES = [
    ("AT-FGSM",         "MLP",     True),
    ("AT-PGD",          "MLP",     True),
    ("Aug-FGSM",        "LogReg",  True),
    ("Aug-Square",      "LogReg",  False),
    ("Aug-FGSM-Iter",   "XGBoost", True),
    ("Aug-Square-Iter", "XGBoost", False),
]
# (défense, modèle, white-box réévalué ?)


def _snake_case(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════

def _load(ds, fname):
    path = RESULTS_DIR / ds / fname
    if not path.exists():
        warnings.warn(f"Fichier de résultats manquant, ignoré : {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_defense_data(ds, defense, model, has_whitebox):
    """→ dict[attack] = (baseline_asr, asr_median, asr_std) pour une défense/modèle/dataset."""
    attacks = (WB_ATTACKS if has_whitebox else []) + BB_ATTACKS + [ENSEMBLE_ATTACK]
    out = {}

    if has_whitebox:
        wb = _load(ds, "defense_results.json")
        wb_entry = (wb or {}).get(model, {}).get(defense)
        if wb_entry is None:
            warnings.warn(f"[{ds}] {model}/{defense} absent de defense_results.json")
        for atk in WB_ATTACKS:
            e = (wb_entry or {}).get(atk)
            if e is None:
                warnings.warn(f"[{ds}] {defense}: attaque {atk} absente (white-box)")
                continue
            out[atk] = (e["baseline_asr"], e["asr_median"], e.get("asr_std", 0.0))

    bb = _load(ds, "defense_results_blackbox.json")
    bb_entry = (bb or {}).get(model, {}).get(defense)
    if bb_entry is None:
        warnings.warn(f"[{ds}] {model}/{defense} absent de defense_results_blackbox.json")
    for atk in BB_ATTACKS + [ENSEMBLE_ATTACK]:
        e = (bb_entry or {}).get(atk)
        if e is None:
            warnings.warn(f"[{ds}] {defense}: attaque {atk} absente (black-box)")
            continue
        out[atk] = (e["baseline_asr"], e["asr_median"], e.get("asr_std", 0.0))

    return attacks, out


# ══════════════════════════════════════════════════════════════
# FIGURE — 2 sous-graphiques empilés (SWaT en haut, BATADAL en bas),
# barres baseline vs défendu par attaque.
# ══════════════════════════════════════════════════════════════

def _plot_defense_panel(ax, attacks, data):
    present = [a for a in attacks if a in data]
    _style_axis(ax)

    if not present:
        ax.text(0.5, 0.5, "Données indisponibles", ha="center", va="center",
                 transform=ax.transAxes, fontsize=10, color="#888888")
        return

    x = np.arange(len(present))
    w = 0.35

    base_vals = [data[a][0] for a in present]
    def_vals  = [data[a][1] for a in present]
    def_errs  = [data[a][2] for a in present]

    ax.bar(x - w / 2, base_vals, width=w * 0.9, color=COLOR_BASELINE,
           edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, label="Baseline (non défendu)",
           zorder=3)
    ax.bar(x + w / 2, def_vals, width=w * 0.9, color=COLOR_DEFENDED,
           edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, label="Modèle défendu", zorder=3)
    ax.errorbar(x + w / 2, def_vals, yerr=def_errs, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=35, ha="right")
    ax.set_ylim(0, 100)


def plot_defense_figure(defense, model, has_whitebox, out_name, figsize=(10, 7)):
    attacks, data_by_ds = None, {}
    for ds, _ in DATASETS:
        attacks, data = load_defense_data(ds, defense, model, has_whitebox)
        data_by_ds[ds] = data

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for ax, (ds, ds_label) in zip(axes, DATASETS):
        _plot_defense_panel(ax, attacks, data_by_ds[ds])
        ax.set_title(ds_label, fontsize=12)
        ax.set_ylabel("ASR médiane (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False)

    fig.suptitle(f"{defense} — ASR avant/après défense ({model}, ε={EPS})",
                 fontsize=13, y=1.06)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("Génération des figures du Chapitre 7...")
    for defense, model, has_whitebox in DEFENSES:
        out_name = f"fig_{_snake_case(defense)}.png"
        plot_defense_figure(defense, model, has_whitebox, out_name)
    print(f"\n-> figures dans {OUT_DIR}")


if __name__ == "__main__":
    main()
