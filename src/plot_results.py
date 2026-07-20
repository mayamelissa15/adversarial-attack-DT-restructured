"""
plot_results.py — Génère les figures (bar charts + boxplots) à partir des CSV
dans results/<dataset>/.

Usage :
    python src/plot_results.py --dataset swat
    python src/plot_results.py --dataset batadal
    python src/plot_results.py --dataset both

Sorties : results/<dataset>/plots/*.png
Tout fichier manquant est simplement ignoré (avec un message), pas d'erreur.
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

from common import results_dir, RESULTS_DIR


# ══════════════════════════════════════════════════════════════
# PALETTE — Okabe-Ito (colorblind-safe), une couleur fixe par modèle,
# utilisée partout de façon cohérente (jamais recyclée pour autre chose).
# ══════════════════════════════════════════════════════════════

MODEL_COLOR = {
    "MLP":     "#0072B2",
    "LogReg":  "#E69F00",
    "XGBoost": "#009E73",
}
MODEL_ORDER = ["MLP", "LogReg", "XGBoost"]
GRID_COLOR  = "#B0B0B0"


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls="--", alpha=0.35, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════
# CHARGEMENT — tolérant aux fichiers manquants
# ══════════════════════════════════════════════════════════════

def _try_read(path):
    if not path.exists():
        print(f"  (absent, ignoré) {path.name}")
        return None
    return pd.read_csv(path)


def load_whitebox_baseline(res_dir, ds, eps):
    dfs = []
    for attack in ["fgsm", "pgd", "cw"]:
        df = _try_read(res_dir / f"whitebox_{attack}_{ds}_eps{eps}.csv")
        if df is not None:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_blackbox_baseline(res_dir, ds, eps, suffix=""):
    dfs = []
    for family in ["score", "transfer", "decision"]:
        df = _try_read(res_dir / f"blackbox_{family}_{ds}_eps{eps}{suffix}.csv")
        if df is not None:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_defense_json(res_dir, name):
    path = res_dir / name
    if not path.exists():
        print(f"  (absent, ignoré) {name}")
        return None
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
# HELPERS — collapse des variantes transfer (substitut) pour la vue
# d'ensemble ; le détail complet reste dispo dans une 2e figure.
# ══════════════════════════════════════════════════════════════

def collapse_transfer_attack(attack):
    if attack.startswith("MI-FGSM"):
        return "MI-FGSM"
    if attack.startswith("VMI-FGSM"):
        return "VMI-FGSM"
    return attack  # Ensemble-MI déjà propre


# ══════════════════════════════════════════════════════════════
# BAR CHART — médiane ± std, groupé par modèle
# ══════════════════════════════════════════════════════════════

def bar_asr(df, attack_order, title, outpath, value_col="asr"):
    agg = (df.groupby(["attack", "model"])[value_col]
             .agg(["median", "std"]).reset_index())

    attacks = [a for a in attack_order if a in agg["attack"].unique()]
    models  = [m for m in MODEL_ORDER if m in agg["model"].unique()]
    if not attacks or not models:
        print(f"  (rien à tracer) {title}")
        return

    n_models = len(models)
    x = np.arange(len(attacks))
    w = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(attacks) * n_models), 4.5))
    _style_axis(ax)

    for i, model in enumerate(models):
        vals, errs = [], []
        for a in attacks:
            row = agg[(agg["attack"] == a) & (agg["model"] == model)]
            vals.append(float(row["median"].iloc[0]) * 100 if not row.empty else np.nan)
            errs.append(float(row["std"].fillna(0).iloc[0]) * 100 if not row.empty else 0)
        xpos = x + (i - n_models / 2 + 0.5) * w
        ax.bar(xpos, vals, width=w * 0.88, color=MODEL_COLOR[model], label=model,
               yerr=errs, capsize=2, zorder=3,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_ylabel("ASR médiane (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  ✓ {outpath.name}")


# ══════════════════════════════════════════════════════════════
# BOXPLOT — distribution de l'ASR à travers les seeds
# ══════════════════════════════════════════════════════════════

def box_asr(df, attack_order, title, outpath, value_col="asr"):
    attacks = [a for a in attack_order if a in df["attack"].unique()]
    models  = [m for m in MODEL_ORDER if m in df["model"].unique()]
    if not attacks or not models:
        print(f"  (rien à tracer) {title}")
        return

    n_models = len(models)
    w = 0.8 / n_models
    x = np.arange(len(attacks))

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(attacks) * n_models), 4.5))
    _style_axis(ax)

    for i, model in enumerate(models):
        data, positions = [], []
        for j, a in enumerate(attacks):
            vals = df[(df["attack"] == a) & (df["model"] == model)][value_col].dropna() * 100
            data.append(vals.values if len(vals) else [np.nan])
            positions.append(x[j] + (i - n_models / 2 + 0.5) * w)

        bp = ax.boxplot(data, positions=positions, widths=w * 0.8, patch_artist=True,
                        showfliers=False, zorder=3)
        for patch in bp["boxes"]:
            patch.set_facecolor(MODEL_COLOR[model])
            patch.set_alpha(0.55)
            patch.set_edgecolor(MODEL_COLOR[model])
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color(MODEL_COLOR[model])

    # légende manuelle (boxplot ne génère pas de handles propres)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLOR[m], alpha=0.55,
                             ec=MODEL_COLOR[m]) for m in models]
    ax.legend(handles, models, frameon=False, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_ylabel("ASR par seed (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  ✓ {outpath.name}")


# ══════════════════════════════════════════════════════════════
# BASELINE vs DÉFENDU — comparaison groupée
# ══════════════════════════════════════════════════════════════

def bar_baseline_vs_defended(df_base, df_def, attack_order, title, outpath):
    attacks = [a for a in attack_order
              if a in df_base["attack"].unique() and a in df_def["attack"].unique()]
    models  = [m for m in MODEL_ORDER if m in df_base["model"].unique()]
    if not attacks or not models:
        print(f"  (rien à tracer) {title}")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    x = np.arange(len(attacks))
    for ax, model in zip(axes, models):
        base_med = [df_base[(df_base.attack == a) & (df_base.model == model)]["asr"].median() * 100
                   for a in attacks]
        def_med  = [df_def[(df_def.attack == a) & (df_def.model == model)]["asr"].median() * 100
                   for a in attacks]
        ax.bar(x - 0.2, base_med, width=0.36, color="#B0B0B0", label="baseline", zorder=3)
        ax.bar(x + 0.2, def_med,  width=0.36, color=MODEL_COLOR[model], label="défendu", zorder=3)
        _style_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=20, ha="right")
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("ASR médiane (%)")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  ✓ {outpath.name}")


# ══════════════════════════════════════════════════════════════
# OVERVIEW ANGLAIS — 3 panneaux (transfer / score / decision) avec
# labels % au-dessus des barres, format article.
# ══════════════════════════════════════════════════════════════

def _label_bars(ax, xpos, vals):
    for xi, v in zip(xpos, vals):
        if np.isnan(v):
            continue
        ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)


def _bar_panel(ax, df, attack_order, title, value_col="asr"):
    agg = (df.groupby(["attack", "model"])[value_col]
             .agg(["median", "std"]).reset_index())
    attacks = [a for a in attack_order if a in agg["attack"].unique()]
    models  = [m for m in MODEL_ORDER if m in agg["model"].unique()]
    if not attacks or not models:
        ax.set_visible(False)
        return

    n_models = len(models)
    x = np.arange(len(attacks))
    w = 0.8 / n_models
    _style_axis(ax)

    ymax = 20.0
    for i, model in enumerate(models):
        vals, errs = [], []
        for a in attacks:
            row = agg[(agg["attack"] == a) & (agg["model"] == model)]
            vals.append(float(row["median"].iloc[0]) * 100 if not row.empty else np.nan)
            errs.append(float(row["std"].fillna(0).iloc[0]) * 100 if not row.empty else 0)
        xpos = x + (i - n_models / 2 + 0.5) * w
        ax.bar(xpos, vals, width=w * 0.88, color=MODEL_COLOR[model], label=model,
               yerr=errs, capsize=2, zorder=3, edgecolor="white", linewidth=0.5)
        _label_bars(ax, xpos, vals)
        ymax = max(ymax, *[v + e for v, e in zip(vals, errs) if not np.isnan(v)])

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_ylim(0, min(ymax + 15, 108))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))


def fig_blackbox_overview(res_dir, ds, eps, out_dir):
    bb = load_blackbox_baseline(res_dir, ds, eps)
    if bb is None:
        print("  (rien à tracer) overview blackbox (en)")
        return

    tr = bb[bb["family"] == "Transfer"].copy()
    sd = bb[bb["family"] == "Score-based"].copy()
    db = bb[bb["family"] == "Decision-based"].copy()
    if tr.empty and sd.empty and db.empty:
        print("  (rien à tracer) overview blackbox (en)")
        return
    tr["attack"] = tr["attack"].apply(collapse_transfer_attack)

    n_tr = tr["seed"].nunique() if not tr.empty else 0
    n_sd = sd["seed"].nunique() if not sd.empty else 0
    n_db = db["seed"].nunique() if not db.empty else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    _bar_panel(axes[0], tr, ["MI-FGSM", "VMI-FGSM", "Ensemble-MI"],
              f"Transfer attacks — Median ASR over {n_tr} runs (ε = {eps})")
    _bar_panel(axes[1], sd, ["Square", "NES"],
              f"Score-based attacks — Median ASR over {n_sd} runs (ε = {eps})")
    _bar_panel(axes[2], db, ["HSJA", "RayS"],
              f"Decision-based attacks — Median ASR over {n_db} runs (ε = {eps})")
    axes[0].set_ylabel("Attack Success Rate (ASR)")

    captions = ["(a) Transfer-based attacks.",
               "(b) Score-based attacks.",
               f"(c) Decision-based attacks ({n_db} runs)."]
    for ax, cap in zip(axes, captions):
        ax.text(0.5, -0.22, cap, transform=ax.transAxes, ha="center", fontsize=9.5)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    outp = out_dir / f"overview_blackbox_{ds}_en.png"
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    print(f"  ✓ {outp.name}")


# ══════════════════════════════════════════════════════════════
# OVERVIEW COMBINÉ — SWaT + BATADAL dans un seul plot (3 panneaux,
# couleur = modèle, hachure = dataset). Pour l'article (limite de place).
# ══════════════════════════════════════════════════════════════

DATASET_HATCH = {"swat": "", "batadal": "///"}
DATASET_LABEL = {"swat": "SWaT", "batadal": "BATADAL"}


def _bar_panel_combined(ax, df, attack_order, title, value_col="asr"):
    agg = (df.groupby(["attack", "model", "dataset"])[value_col]
             .agg(["median", "std"]).reset_index())
    attacks  = [a for a in attack_order if a in agg["attack"].unique()]
    models   = [m for m in MODEL_ORDER if m in agg["model"].unique()]
    datasets = [d for d in ["swat", "batadal"] if d in agg["dataset"].unique()]
    if not attacks or not models or not datasets:
        ax.set_visible(False)
        return

    _style_axis(ax)
    n_bars = len(models) * len(datasets)
    x = np.arange(len(attacks))
    w = 0.85 / n_bars

    ymax = 20.0
    bar_idx = 0
    for model in models:
        for ds in datasets:
            vals, errs = [], []
            for a in attacks:
                row = agg[(agg["attack"] == a) & (agg["model"] == model) & (agg["dataset"] == ds)]
                vals.append(float(row["median"].iloc[0]) * 100 if not row.empty else np.nan)
                errs.append(float(row["std"].fillna(0).iloc[0]) * 100 if not row.empty else 0)
            xpos = x + (bar_idx - n_bars / 2 + 0.5) * w
            ax.bar(xpos, vals, width=w * 0.92, color=MODEL_COLOR[model],
                   hatch=DATASET_HATCH[ds], edgecolor="white", linewidth=0.5,
                   yerr=errs, capsize=1.5, zorder=3)
            for xi, v in zip(xpos, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
            finite = [v + e for v, e in zip(vals, errs) if not np.isnan(v)]
            if finite:
                ymax = max(ymax, *finite)
            bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, min(ymax + 22, 112))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))


def fig_blackbox_overview_combined(eps, datasets=("swat", "batadal")):
    print(f"\n{'═'*60}\n  PLOT COMBINÉ — {' + '.join(d.upper() for d in datasets)}\n{'═'*60}")

    dfs = []
    for ds in datasets:
        bb = load_blackbox_baseline(results_dir(ds), ds, eps)
        if bb is not None:
            dfs.append(bb)
    if not dfs:
        print("  (rien à tracer) overview combiné")
        return
    bb = pd.concat(dfs, ignore_index=True)

    tr = bb[bb["family"] == "Transfer"].copy()
    sd = bb[bb["family"] == "Score-based"].copy()
    db = bb[bb["family"] == "Decision-based"].copy()
    tr["attack"] = tr["attack"].apply(collapse_transfer_attack)

    n_tr = tr["seed"].nunique() if not tr.empty else 0
    n_sd = sd["seed"].nunique() if not sd.empty else 0
    n_db = db["seed"].nunique() if not db.empty else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    _bar_panel_combined(axes[0], tr, ["MI-FGSM", "VMI-FGSM", "Ensemble-MI"],
                        f"Transfer attacks — Median ASR over {n_tr} runs (ε = {eps})")
    _bar_panel_combined(axes[1], sd, ["Square", "NES"],
                        f"Score-based attacks — Median ASR over {n_sd} runs (ε = {eps})")
    _bar_panel_combined(axes[2], db, ["HSJA", "RayS"],
                        f"Decision-based attacks — Median ASR over {n_db} runs (ε = {eps})")
    axes[0].set_ylabel("Attack Success Rate (ASR)")

    captions = ["(a) Transfer-based attacks.",
               "(b) Score-based attacks.",
               f"(c) Decision-based attacks ({n_db} runs)."]
    for ax, cap in zip(axes, captions):
        ax.text(0.5, -0.20, cap, transform=ax.transAxes, ha="center", fontsize=9.5)

    model_handles = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLOR[m], ec="none") for m in MODEL_ORDER]
    dataset_handles = [plt.Rectangle((0, 0), 1, 1, fc="white", ec="black",
                                     hatch=DATASET_HATCH[d]) for d in datasets]
    fig.legend(model_handles + dataset_handles,
              MODEL_ORDER + [DATASET_LABEL[d] for d in datasets],
              loc="upper center", ncol=len(MODEL_ORDER) + len(datasets),
              frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.10))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_dir = RESULTS_DIR / "combined" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / "overview_blackbox_all_en.png"
    fig.savefig(outp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {outp}")


# ══════════════════════════════════════════════════════════════
# OVERVIEW DÉFENSES COMBINÉ — SWaT + BATADAL, 1 panneau par modèle
# (baseline vs. défenses restantes), hachure = dataset. Défenses
# à base de Square exclues (résultats contre-productifs).
# ══════════════════════════════════════════════════════════════

FAMILY_COLOR = {"Baseline": "#B0B0B0", "FGSM": "#D55E00", "PGD": "#7570B3"}
FAMILY_LABEL = {"Baseline": "Baseline", "FGSM": "FGSM-based defense", "PGD": "PGD-based defense"}


def _defense_family(name):
    if name == "Baseline":
        return "Baseline"
    return "PGD" if "PGD" in name else "FGSM"


def _bar_panel_defense(ax, df, attack_order, title, datasets, series_order):
    attacks = [a for a in attack_order if a in df["attack"].unique()]
    if not attacks or not series_order:
        ax.set_visible(False)
        return

    _style_axis(ax)
    n_bars = len(series_order) * len(datasets)
    x = np.arange(len(attacks))
    w = 0.85 / n_bars

    ymax = 20.0
    bar_idx = 0
    for s in series_order:
        color = FAMILY_COLOR[_defense_family(s)]
        for ds in datasets:
            vals = []
            for a in attacks:
                row = df[(df["attack"] == a) & (df["series"] == s) & (df["dataset"] == ds)]
                vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
            xpos = x + (bar_idx - n_bars / 2 + 0.5) * w
            ax.bar(xpos, vals, width=w * 0.92, color=color, hatch=DATASET_HATCH[ds],
                  edgecolor="white", linewidth=0.5, zorder=3)
            finite = [v for v in vals if not np.isnan(v)]
            if finite:
                ymax = max(ymax, *finite)
            bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=8.5, rotation=20, ha="right")
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, min(ymax + 15, 108))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))


def fig_defense_overview_combined(eps, datasets=("swat", "batadal"), exclude_substr="square"):
    print(f"\n{'═'*60}\n  PLOT DÉFENSES COMBINÉ — {' + '.join(d.upper() for d in datasets)}\n{'═'*60}")

    rows = []
    for ds in datasets:
        data = load_defense_json(results_dir(ds), "defense_results_blackbox.json")
        if not data:
            continue
        for base_model, defenses in data.items():
            for defense, attacks in defenses.items():
                if exclude_substr in defense.lower():
                    continue
                for attack, m in attacks.items():
                    attack_c = collapse_transfer_attack(attack)
                    rows.append({"dataset": ds, "base_model": base_model, "series": defense,
                                "attack": attack_c, "value": m["asr_median"]})
                    rows.append({"dataset": ds, "base_model": base_model, "series": "Baseline",
                                "attack": attack_c, "value": m["baseline_asr"]})
    if not rows:
        print("  (rien à tracer) overview défenses")
        return
    df = (pd.DataFrame(rows)
            .groupby(["dataset", "base_model", "series", "attack"], as_index=False)["value"].mean())

    attack_order = ["MI-FGSM", "VMI-FGSM", "Ensemble-MI", "Square", "NES", "HSJA", "RayS"]
    models = [m for m in MODEL_ORDER if m in df["base_model"].unique()]
    if not models:
        print("  (rien à tracer) overview défenses")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6))
    if len(models) == 1:
        axes = [axes]

    for ax, bm in zip(axes, models):
        sub = df[df["base_model"] == bm]
        series_order = sorted(sub["series"].unique(),
                              key=lambda s: (0 if s == "Baseline" else (2 if "PGD" in s else 1), s))
        _bar_panel_defense(ax, sub, attack_order,
                          f"{bm} — baseline vs. defenses (ε = {eps})", datasets, series_order)
    axes[0].set_ylabel("Attack Success Rate (ASR)")

    captions = [f"({chr(97 + i)}) {bm}." for i, bm in enumerate(models)]
    for ax, cap in zip(axes, captions):
        ax.text(0.5, -0.30, cap, transform=ax.transAxes, ha="center", fontsize=9.5)

    families_present = sorted({_defense_family(s) for s in df["series"].unique()},
                              key=lambda f: {"Baseline": 0, "FGSM": 1, "PGD": 2}[f])
    color_handles = [plt.Rectangle((0, 0), 1, 1, fc=FAMILY_COLOR[f], ec="none") for f in families_present]
    dataset_handles = [plt.Rectangle((0, 0), 1, 1, fc="white", ec="black",
                                     hatch=DATASET_HATCH[d]) for d in datasets]
    fig.legend(color_handles + dataset_handles,
              [FAMILY_LABEL[f] for f in families_present] + [DATASET_LABEL[d] for d in datasets],
              loc="upper center", ncol=len(families_present) + len(datasets),
              frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.12))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out_dir = RESULTS_DIR / "combined" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / "overview_defenses_all_en.png"
    fig.savefig(outp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {outp}")


# ══════════════════════════════════════════════════════════════
# MAIN — un dataset
# ══════════════════════════════════════════════════════════════

def run_dataset(ds, eps):
    res_dir = results_dir(ds)
    out_dir = res_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'═'*60}\n  PLOTS — {ds.upper()}\n{'═'*60}")

    # ── Whitebox baseline ──
    wb = load_whitebox_baseline(res_dir, ds, eps)
    if wb is not None:
        bar_asr(wb, ["FGSM", "PGD", "C&W"],
               f"Whitebox baseline — {ds.upper()} (eps={eps})",
               out_dir / "whitebox_baseline_bar.png")
        box_asr(wb, ["FGSM", "PGD", "C&W"],
               f"Whitebox baseline — distribution par seed — {ds.upper()}",
               out_dir / "whitebox_baseline_box.png")

    # ── Blackbox baseline : score + decision ──
    bb = load_blackbox_baseline(res_dir, ds, eps)
    if bb is not None:
        sd = bb[bb["attack"].isin(["Square", "NES", "HSJA", "RayS"])]
        if not sd.empty:
            bar_asr(sd, ["Square", "NES", "HSJA", "RayS"],
                   f"Blackbox score/decision baseline — {ds.upper()} (eps={eps})",
                   out_dir / "blackbox_score_decision_bar.png")
            box_asr(sd, ["Square", "NES", "HSJA", "RayS"],
                   f"Blackbox score/decision — distribution par seed — {ds.upper()}",
                   out_dir / "blackbox_score_decision_box.png")

        tr = bb[bb["family"] == "Transfer"].copy()
        if not tr.empty:
            tr["attack_base"] = tr["attack"].apply(collapse_transfer_attack)
            tr_overview = tr.rename(columns={"attack": "attack_detail"}) \
                            .rename(columns={"attack_base": "attack"})
            bar_asr(tr_overview, ["MI-FGSM", "VMI-FGSM", "Ensemble-MI"],
                   f"Blackbox transfer (agrégé substituts) — {ds.upper()} (eps={eps})",
                   out_dir / "blackbox_transfer_bar.png")
            box_asr(tr_overview, ["MI-FGSM", "VMI-FGSM", "Ensemble-MI"],
                   f"Blackbox transfer — distribution par seed — {ds.upper()}",
                   out_dir / "blackbox_transfer_box.png")

    # ── Overview anglais 3 panneaux (transfer/score/decision), style article ──
    fig_blackbox_overview(res_dir, ds, eps, out_dir)

    # ── Blackbox défendu vs baseline ──
    bb_def = load_blackbox_baseline(res_dir, ds, eps, suffix="_defended")
    if bb is not None and bb_def is not None:
        sd_base = bb[bb["attack"].isin(["Square", "NES", "HSJA", "RayS"])]
        sd_def  = bb_def[bb_def["attack"].isin(["Square", "NES", "HSJA", "RayS"])]
        if not sd_base.empty and not sd_def.empty:
            bar_baseline_vs_defended(sd_base, sd_def, ["Square", "NES", "HSJA", "RayS"],
                                    f"Blackbox baseline vs défendu (MLP durci) — {ds.upper()}",
                                    out_dir / "blackbox_baseline_vs_defended_bar.png")

    # ── Matrice défenses (evaluate.py / evaluate_blackbox.py) ──
    for name, label in [("defense_results.json", "whitebox"),
                        ("defense_results_blackbox.json", "blackbox")]:
        data = load_defense_json(res_dir, name)
        if not data:
            continue
        rows = []
        for base_model, defenses in data.items():
            for defense, attacks in defenses.items():
                for attack, m in attacks.items():
                    rows.append({"base_model": base_model, "defense": defense,
                                "attack": attack, "asr_median": m["asr_median"],
                                "delta_asr": m["delta_asr"]})
        ddf = pd.DataFrame(rows)
        if ddf.empty:
            continue
        for bm in ddf["base_model"].unique():
            sub = ddf[ddf["base_model"] == bm]
            fig, ax = plt.subplots(figsize=(max(6, 1.3 * sub["attack"].nunique()), 4.5))
            _style_axis(ax)
            defenses = sorted(sub["defense"].unique())
            attacks  = sorted(sub["attack"].unique())
            x = np.arange(len(attacks))
            n = len(defenses)
            w = 0.8 / max(n, 1)
            cmap = plt.cm.get_cmap("tab10")
            for i, defense in enumerate(defenses):
                vals = [sub[(sub.attack == a) & (sub.defense == defense)]["asr_median"]
                       .mean() for a in attacks]
                ax.bar(x + (i - n / 2 + 0.5) * w, vals, width=w * 0.9,
                      label=defense, color=cmap(i % 10), zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(attacks, rotation=25, ha="right")
            ax.set_ylabel("ASR médiane (%)")
            ax.set_ylim(0, 105)
            ax.set_title(f"Défenses {label} — {bm} — {ds.upper()}", fontsize=10, fontweight="bold")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            outp = out_dir / f"defense_{label}_{bm}.png"
            fig.savefig(outp, dpi=150)
            plt.close(fig)
            print(f"  ✓ {outp.name}")

    print(f"\n  → figures dans {out_dir}")


# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Génère les plots depuis results/")
    p.add_argument("--dataset", default="both", choices=["swat", "batadal", "both"])
    p.add_argument("--eps", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    datasets = ["swat", "batadal"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, args.eps)
    if args.dataset == "both":
        fig_blackbox_overview_combined(args.eps, datasets=tuple(datasets))
        fig_defense_overview_combined(args.eps, datasets=tuple(datasets))


if __name__ == "__main__":
    main()
