"""
generate_figures_chapitre6.py — Figures du Chapitre 6 (résultats d'attaque)
du mémoire, regroupées par famille d'attaque.

Lit DIRECTEMENT les JSON agrégés (déjà multi-seed : médiane + écart-type)
dans results/<dataset>/ :
    whitebox_fgsm_<ds>_eps0.1.json
    whitebox_pgd_<ds>_eps0.1.json
    whitebox_cw_<ds>_eps0.1.json
    blackbox_score_<ds>_eps0.1.json
    blackbox_decision_<ds>_eps0.1.json
    blackbox_transfer_<ds>_eps0.1.json

Ne recalcule et ne relance rien : lecture seule de results/. Si un fichier
attendu est absent, un warning explicite est loggé et la figure concernée
est générée avec les données restantes (voire ignorée si aucune donnée).

Sortie : figures/chapitre6/fig_whitebox.png, fig_score_based.png,
         fig_decision_based.png, fig_transfert.png

Usage :
    python src/generate_figures_chapitre6.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR     = ROOT / "figures" / "chapitre6"
EPS         = 0.1
DATASETS    = [("swat", "SWaT"), ("batadal", "BATADAL")]

# ══════════════════════════════════════════════════════════════
# PALETTE — pastel, une couleur par modèle, réutilisée sans exception
# ══════════════════════════════════════════════════════════════

MODEL_COLOR = {
    "MLP":     "#A7C7E7",  # bleu pastel
    "LogReg":  "#F6D186",  # jaune/orange pastel
    "XGBoost": "#F4A6A6",  # rose/rouge pastel
}
MODEL_ORDER = ["MLP", "LogReg", "XGBoost"]

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
# CHARGEMENT
# ══════════════════════════════════════════════════════════════

def _load(ds, fname):
    path = RESULTS_DIR / ds / fname
    if not path.exists():
        warnings.warn(f"Fichier de résultats manquant, ignoré : {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_whitebox(ds):
    """→ dict[model][attack] = (median, std) pour FGSM / PGD / C&W."""
    out = {}
    for fname, attack_key in [
        (f"whitebox_fgsm_{ds}_eps{EPS}.json", "FGSM"),
        (f"whitebox_pgd_{ds}_eps{EPS}.json",  "PGD"),
        (f"whitebox_cw_{ds}_eps{EPS}.json",   "C&W"),
    ]:
        data = _load(ds, fname)
        if data is None:
            continue
        for model, attacks in data.items():
            m = attacks.get(attack_key)
            if m is None:
                warnings.warn(f"[{ds}] {attack_key} absent pour {model} dans {fname}")
                continue
            out.setdefault(model, {})[attack_key] = (
                m["evasion_rate_median"], m.get("evasion_rate_std", 0.0)
            )
    return out


def load_blackbox_family(ds, family):
    """family in {'score', 'decision', 'transfer'} → dict[model][attack] = (median, std)."""
    fname = f"blackbox_{family}_{ds}_eps{EPS}.json"
    data = _load(ds, fname)
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
# FIGURE GÉNÉRIQUE — 2 sous-graphiques (SWaT | BATADAL), barres
# groupées par attaque, une couleur par modèle. Réutilisée pour
# white-box / score-based / decision-based.
# ══════════════════════════════════════════════════════════════

def _plot_grouped_panel(ax, data, attack_order):
    models  = [m for m in MODEL_ORDER if m in data]
    attacks = [a for a in attack_order if any(a in data.get(m, {}) for m in models)]
    _style_axis(ax)

    if not models or not attacks:
        ax.text(0.5, 0.5, "Données indisponibles", ha="center", va="center",
                 transform=ax.transAxes, fontsize=10, color="#888888")
        return

    n_models = len(models)
    x = np.arange(len(attacks))
    w = 0.78 / n_models

    for i, model in enumerate(models):
        vals, errs = [], []
        for a in attacks:
            med, std = data.get(model, {}).get(a, (np.nan, 0.0))
            vals.append(med)
            errs.append(std)
        xpos = x + (i - n_models / 2 + 0.5) * w
        ax.bar(xpos, vals, width=w * 0.9, color=MODEL_COLOR[model], label=model,
               edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, zorder=3)
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="black",
                     elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylim(0, 100)


def plot_attack_family_figure(attack_order, data_by_dataset, suptitle, out_name,
                               figsize=(11, 4.5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, (ds, ds_label) in zip(axes, DATASETS):
        _plot_grouped_panel(ax, data_by_dataset.get(ds, {}), attack_order)
        ax.set_title(ds_label, fontsize=12)

    axes[0].set_ylabel("ASR médiane (%)")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR[m],
                              edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, label=m)
               for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(MODEL_ORDER), frameon=False)

    fig.suptitle(suptitle, fontsize=13, y=1.1)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out_path}")


# ══════════════════════════════════════════════════════════════
# FIGURE TRANSFERT — structure dédiée : 7 séries (3 substituts x
# {MI-FGSM, VMI-FGSM} + Ensemble-MI), Ensemble-MI visuellement isolé.
# ══════════════════════════════════════════════════════════════

TRANSFER_SERIES = [
    ("MI-FGSM_Sub1-MLP",       "MI-FGSM\nSub1"),
    ("VMI-FGSM_Sub1-MLP",      "VMI-FGSM\nSub1"),
    ("MI-FGSM_Sub2-SmallMLP",  "MI-FGSM\nSub2"),
    ("VMI-FGSM_Sub2-SmallMLP", "VMI-FGSM\nSub2"),
    ("MI-FGSM_Sub3-DeepMLP",   "MI-FGSM\nSub3"),
    ("VMI-FGSM_Sub3-DeepMLP",  "VMI-FGSM\nSub3"),
]
ENSEMBLE_KEY   = "Ensemble-MI"
ENSEMBLE_LABEL = "Ensemble-MI"


def _plot_transfer_panel(ax, data):
    models = [m for m in MODEL_ORDER if m in data]
    _style_axis(ax)

    if not models:
        ax.text(0.5, 0.5, "Données indisponibles", ha="center", va="center",
                 transform=ax.transAxes, fontsize=10, color="#888888")
        return

    n_series = len(TRANSFER_SERIES) + 1   # 6 substituts + Ensemble-MI
    n_models = len(models)
    x = np.arange(n_series)
    extra_gap = 0.6   # espace supplémentaire avant Ensemble-MI
    x[-1] += extra_gap
    w = 0.78 / n_models

    # fond légèrement grisé derrière la colonne Ensemble-MI
    ens_center = x[-1]
    ax.axvspan(ens_center - 0.5 - extra_gap / 2, ens_center + 0.5 + extra_gap / 2,
               color="#F0F0F0", zorder=0)

    series_keys = [k for k, _ in TRANSFER_SERIES] + [ENSEMBLE_KEY]
    labels = [lbl for _, lbl in TRANSFER_SERIES] + [ENSEMBLE_LABEL]

    ymax = 10.0
    for i, model in enumerate(models):
        vals, errs = [], []
        for key in series_keys:
            med, std = data.get(model, {}).get(key, (np.nan, 0.0))
            vals.append(med)
            errs.append(std)
            if not np.isnan(med):
                ymax = max(ymax, med + std)
        xpos = x + (i - n_models / 2 + 0.5) * w
        ax.bar(xpos, vals, width=w * 0.9, color=MODEL_COLOR[model], label=model,
               edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, zorder=3)
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="black",
                     elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 100)


def plot_transfer_figure(data_by_dataset, out_name="fig_transfert.png",
                          figsize=(13, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, (ds, ds_label) in zip(axes, DATASETS):
        _plot_transfer_panel(ax, data_by_dataset.get(ds, {}))
        ax.set_title(ds_label, fontsize=12)

    axes[0].set_ylabel("ASR médiane (%)")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR[m],
                              edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, label=m)
               for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(MODEL_ORDER), frameon=False)

    fig.suptitle("Attaques par transfert — ASR médiane par modèle (ε=0.1)",
                 fontsize=13, y=1.12)
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
    whitebox = {ds: load_whitebox(ds) for ds, _ in DATASETS}
    score    = {ds: load_blackbox_family(ds, "score") for ds, _ in DATASETS}
    decision = {ds: load_blackbox_family(ds, "decision") for ds, _ in DATASETS}
    transfer = {ds: load_blackbox_family(ds, "transfer") for ds, _ in DATASETS}

    print("Génération des figures du Chapitre 6...")

    plot_attack_family_figure(
        ["FGSM", "PGD", "C&W"], whitebox,
        "Attaques white-box — ASR médiane par modèle (ε=0.1)",
        "fig_whitebox.png",
    )
    plot_attack_family_figure(
        ["Square", "NES"], score,
        "Attaques black-box score-based — ASR médiane par modèle (ε=0.1)",
        "fig_score_based.png",
    )
    plot_attack_family_figure(
        ["HSJA", "RayS"], decision,
        "Attaques black-box decision-based — ASR médiane par modèle (ε=0.1)",
        "fig_decision_based.png",
    )
    plot_transfer_figure(transfer)

    print(f"\n-> figures dans {OUT_DIR}")


if __name__ == "__main__":
    main()
