"""
plot_chapitre6.py — Figures du Chapitre 6 (résultats de défense) du mémoire.

Lit DIRECTEMENT les JSON agrégés dans results/<dataset>/ :
    defense_results.json            (défenses vs. attaques white-box)
    defense_results_blackbox.json   (défenses vs. attaques black-box)

AT-Square (MLP) est explicitement HORS PÉRIMÈTRE et exclu de tous les
graphiques (variable EXCLUDE_MODEL_DEFENSE ci-dessous).

Sortie : figures/chapitre6/*.png — PNG haute résolution (dpi=300), pas de
titre dans la figure (le \\caption LaTeX s'en charge).

Usage :
    python src/plot_chapitre6.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR     = ROOT / "figures" / "chapitre6"

# ══════════════════════════════════════════════════════════════
# STYLE — cohérent avec le Chapitre 5
# ══════════════════════════════════════════════════════════════

MODEL_COLOR = {
    "MLP":     "#0072B2",
    "LogReg":  "#E69F00",
    "XGBoost": "#009E73",
}
MODEL_ORDER  = ["MLP", "LogReg", "XGBoost"]
GRID_COLOR   = "#CCCCCC"
ALPHA_BASE   = 0.35   # baseline (avant défense) : couleur claire
ALPHA_DEF    = 1.0    # défendu : couleur pleine

plt.rcParams.update({
    "font.size":        9,
    "axes.labelsize":   9.5,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "font.family":      "sans-serif",
    "savefig.dpi":      300,
    "pdf.fonttype":     42,
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
# EXCLUSIONS — AT-Square (MLP) hors périmètre du mémoire
# ══════════════════════════════════════════════════════════════

EXCLUDE_MODEL_DEFENSE = {("MLP", "AT-Square")}


# ══════════════════════════════════════════════════════════════
# CHARGEMENT / EXTRACTION
# ══════════════════════════════════════════════════════════════

def _load(ds, fname):
    path = RESULTS_DIR / ds / fname
    if not path.exists():
        print(f"  (absent, ignoré) {fname}")
        return None
    with open(path) as f:
        return json.load(f)


def extract(raw, model_defense_map, attack_order):
    """
    raw : dict brut {model: {defense: {attack: {asr_median, asr_std,
          baseline_asr, ...}}}}
    model_defense_map : dict model -> liste ordonnée de défenses à tracer
          (1re = pas de hachure, 2e = hachure '//')
    → dict model -> {
          "baseline": {attack: median},
          "defenses": {defense: {attack: (median, std)}}
      }
    """
    out = {}
    for model, defenses in model_defense_map.items():
        if raw is None or model not in raw:
            continue
        model_out = {"baseline": {}, "defenses": {}}
        for defense in defenses:
            if (model, defense) in EXCLUDE_MODEL_DEFENSE:
                continue
            attack_dict = raw[model].get(defense)
            if attack_dict is None:
                continue
            series = {}
            for attack in attack_order:
                m = attack_dict.get(attack)
                if m is None:
                    continue
                series[attack] = (m["asr_median"], m.get("asr_std", 0.0))
                model_out["baseline"].setdefault(attack, m["baseline_asr"])
            if series:
                model_out["defenses"][defense] = series
        if model_out["defenses"]:
            out[model] = model_out
    return out


# ══════════════════════════════════════════════════════════════
# TRACÉ — 1 subplot par modèle, groupé par attaque, baseline (clair)
# + défense(s) (plein, hachure si >1 défense pour ce modèle)
# ══════════════════════════════════════════════════════════════

HATCH_BY_RANK = ["", "//"]   # 1re défense = pleine, 2e = hachurée


def _plot_model_panel(ax, model, model_data, attack_order):
    attacks = [a for a in attack_order if a in model_data["baseline"]]
    defenses = list(model_data["defenses"].keys())
    n_series = 1 + len(defenses)
    x = np.arange(len(attacks))
    w = 0.82 / n_series
    color = MODEL_COLOR[model]

    _style_axis(ax)

    # ── baseline ──
    vals = [model_data["baseline"][a] for a in attacks]
    xpos = x + (0 - n_series / 2 + 0.5) * w
    ax.bar(xpos, vals, width=w * 0.9, color=color, alpha=ALPHA_BASE,
           edgecolor="black", linewidth=0.5, zorder=3)

    # ── défense(s) ──
    for i, defense in enumerate(defenses):
        series = model_data["defenses"][defense]
        vals, errs = [], []
        for a in attacks:
            med, std = series.get(a, (np.nan, 0.0))
            vals.append(med)
            errs.append(std)
        xpos = x + (i + 1 - n_series / 2 + 0.5) * w
        ax.bar(xpos, vals, width=w * 0.9, color=color, alpha=ALPHA_DEF,
               edgecolor="black", linewidth=0.5, hatch=HATCH_BY_RANK[i], zorder=3)
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=15, ha="right")
    ax.set_xlabel(model, fontsize=10, fontweight="bold", labelpad=8)

    all_vals = [model_data["baseline"][a] for a in attacks]
    for defense in defenses:
        for a in attacks:
            med, std = model_data["defenses"][defense].get(a, (np.nan, 0.0))
            if not np.isnan(med):
                all_vals.append(med + std)
    ymax = max(all_vals) if all_vals else 10.0
    ax.set_ylim(0, min(ymax * 1.22, 112))

    # légende locale (hachure = nom de défense), seulement si >1 défense
    if len(defenses) > 1:
        handles = [Patch(facecolor="white", edgecolor="black", hatch=h, label=d)
                  for d, h in zip(defenses, HATCH_BY_RANK)]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8,
                 handlelength=1.4, handletextpad=0.5, borderaxespad=0.2)


def defense_figure(data, attack_order, out_path):
    models = [m for m in MODEL_ORDER if m in data]
    if not models:
        print(f"  (rien à tracer) {out_path.name}")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.0), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        _plot_model_panel(ax, model, data[model], attack_order)
    axes[0].set_ylabel("ASR médiane (%)")

    # légende globale : "Modèle — baseline" / "Modèle — défendu", 2 par modèle
    handles = []
    for model in models:
        color = MODEL_COLOR[model]
        handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.5,
                             alpha=ALPHA_BASE, label=f"{model} — baseline"))
        handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.5,
                             alpha=ALPHA_DEF, label=f"{model} — défendu"))
    fig.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), ncol=len(models), fontsize=8.7,
              handlelength=1.3, columnspacing=1.3, handletextpad=0.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ══════════════════════════════════════════════════════════════
# CONFIG PAR FIGURE
# ══════════════════════════════════════════════════════════════

WHITEBOX_DEFENSES = {
    "MLP":     ["AT-FGSM", "AT-PGD"],
    "LogReg":  ["Aug-FGSM"],
    "XGBoost": ["Aug-FGSM-Iter"],
}
WHITEBOX_ATTACKS = ["FGSM", "PGD", "C&W"]

BLACKBOX_DEFENSES = {
    "MLP":     ["AT-FGSM", "AT-PGD"],          # AT-Square exclu (hors périmètre)
    "LogReg":  ["Aug-FGSM", "Aug-Square"],
    "XGBoost": ["Aug-FGSM-Iter", "Aug-Square-Iter"],
}
BLACKBOX_ATTACKS = ["Square", "NES", "HSJA", "RayS", "Ensemble-MI"]


def run_dataset(ds):
    print(f"\n{'='*60}\n  CHAPITRE 6 - {ds.upper()}\n{'='*60}")

    raw_wb = _load(ds, "defense_results.json")
    data_wb = extract(raw_wb, WHITEBOX_DEFENSES, WHITEBOX_ATTACKS)
    defense_figure(data_wb, WHITEBOX_ATTACKS, OUT_DIR / f"fig_defense_whitebox_{ds}.png")

    raw_bb = _load(ds, "defense_results_blackbox.json")
    data_bb = extract(raw_bb, BLACKBOX_DEFENSES, BLACKBOX_ATTACKS)
    defense_figure(data_bb, BLACKBOX_ATTACKS, OUT_DIR / f"fig_defense_blackbox_{ds}.png")


def main():
    for ds in ["batadal", "swat"]:
        run_dataset(ds)
    print(f"\n-> figures dans {OUT_DIR}")


if __name__ == "__main__":
    main()
