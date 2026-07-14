"""
blackbox_run.py — Runner BLACK-BOX unifié
══════════════════════════════════════════

Remplace blackbox_multirun.py. Trois familles :
  - Score-based    : Square, NES
  - Transfer       : MI-FGSM, VMI-FGSM, Ensemble-MI (3 substituts)
  - Decision-based : HSJA, RayS  (longs → checkpoint/reprise)

Nouveautés vs ancienne version :
  - Chemins via common (artifacts/results). Seuil = common.THRESHOLD (0.5).
  - Même build_per_model_eval que le white-box → eval sets identiques,
    ASR comparables entre familles ET entre white/black-box.
  - Timestamps intégrés (--timestamps auto|on|off) : export par-échantillon.
  - --victims baseline|defended : en mode 'defended', RÉ-ATTAQUE les modèles
    durcis (= évaluation adaptative des défenses côté black-box).
  - Sorties par famille (pas de collision en parallèle).

Usage :
  # baseline, tout en un
  python src/blackbox_run.py --dataset swat --eps 0.1
  # en parallèle (une famille par fenêtre tmux)
  python src/blackbox_run.py --dataset swat --eps 0.1 --only score
  python src/blackbox_run.py --dataset swat --eps 0.1 --only transfer
  python src/blackbox_run.py --dataset swat --eps 0.1 --only decision
  # défense (ré-attaque les modèles durcis)
  python src/blackbox_run.py --dataset swat --eps 0.1 --victims defended

Sorties : results/<ds>/blackbox_<family>_<ds>_eps<eps>[_defended].csv/.json
          (+ blackbox_persample_<family>_... si timestamps)
"""

import argparse
import json
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from common import (SEED, THRESHOLD, set_all_seeds, artifacts_dir, results_dir,
                    eval_attack, eval_attack_persample, subsample_for_substitute)
from common_whitebox import (get_device, eval_sizes, build_per_model_eval,
                             load_victims, load_victims_defended, load_timestamps)
from models import MLP, SmallMLP, DeepMLP
from blackbox import square_attack, nes_attack, hsja, rays
from transfer import mi_fgsm, vmi_fgsm, ensemble_mi_fgsm, train_substitute, eval_transfer


# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Runner black-box unifié")
    p.add_argument("--dataset", required=True, choices=["swat", "batadal"])
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--only", default=None, choices=["score", "transfer", "decision"])
    p.add_argument("--skip_transfer", action="store_true")
    p.add_argument("--victims", default="baseline", choices=["baseline", "defended"])
    p.add_argument("--mlp-defense", default="at_pgd", choices=["at_pgd", "at_fgsm"])
    p.add_argument("--timestamps", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--persample_n", default=50, type=int)
    p.add_argument("--fast", action="store_true",
                   help="Réduit le coût decision-based (pour tester la chaîne)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def already_done(df, seed, attack, model):
    if df is None or df.empty:
        return False
    return not df[(df["seed"] == seed) & (df["attack"] == attack) &
                  (df["model"] == model)].empty


def subsample_bb(X_atk, y_atk, seed, max_n):
    """Sous-échantillonne les attaques pour le decision-based (coûteux)."""
    if len(X_atk) <= max_n:
        return X_atk, y_atk, np.arange(len(X_atk))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_atk), max_n, replace=False)
    return X_atk[idx], y_atk[idx], idx


def build_bb_eval(X_eval, y_eval, X_atk_bb):
    """Eval set decision-based = normaux + attaques sous-échantillonnées."""
    m = (y_eval == 0)
    X_bb = np.concatenate([X_eval[m], X_atk_bb], axis=0)
    y_bb = np.concatenate([y_eval[m], np.ones(len(X_atk_bb), dtype=y_eval.dtype)], axis=0)
    return X_bb, y_bb


def add_persample(store, wrapper, X_full, y_full, X_adv, label, model,
                  ts_full, persample_n, seed, eps, ds):
    recs = eval_attack_persample(wrapper, X_full, y_full, X_adv, label, model,
                                 timestamps_full=ts_full, threshold=THRESHOLD,
                                 max_samples=persample_n, seed=seed)
    for r in recs:
        r.update({"eps": eps, "dataset": ds})
    store.extend(recs)


def write_family(family, rows, ps_rows, res_dir, tag, do_ts):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / f"blackbox_{family}_{tag}.csv", index=False)
    print(f"\n✓ CSV → blackbox_{family}_{tag}.csv")

    summary = (df.groupby(["attack", "model"])["asr"]
                 .agg(["median", "std", "min", "max"]).round(4))
    print(f"  RÉSUMÉ {family} — {tag}")
    print(summary.to_string())

    out = {}
    for model in df["model"].unique():
        out[model] = {}
        for attack in df[df["model"] == model]["attack"].unique():
            vals = df[(df["model"] == model) & (df["attack"] == attack)]["asr"]
            out[model][attack] = {
                "evasion_rate_median": round(float(vals.median()) * 100, 2),
                "evasion_rate_std":    round(float(vals.std(ddof=0)) * 100, 2),
                "evasion_rate_min":    round(float(vals.min()) * 100, 2),
                "evasion_rate_max":    round(float(vals.max()) * 100, 2),
            }
    with open(res_dir / f"blackbox_{family}_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)

    if do_ts and ps_rows:
        pd.DataFrame(ps_rows).to_csv(
            res_dir / f"blackbox_persample_{family}_{tag}.csv", index=False)


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    DS, EPS, N_RUNS = args.dataset, args.eps, args.n_runs
    SAVE, RES, DEVICE = artifacts_dir(DS), results_dir(DS), get_device()
    SEEDS = list(range(N_RUNS))
    EVAL_ATK, EVAL_NRM = eval_sizes(DS)
    PN = args.persample_n

    suffix = "_defended" if args.victims == "defended" else ""
    TAG = f"{DS}_eps{EPS}{suffix}"

    RUN_SCORE    = args.only in (None, "score")
    RUN_TRANSFER = args.only in (None, "transfer") and not args.skip_transfer
    RUN_DECISION = args.only in (None, "decision")

    # Hyperparams decision-based (réduits en --fast)
    MAX_DB      = 100 if args.fast else 300
    HSJA_ITERS  = 10  if args.fast else 40
    HSJA_NEST   = 30  if args.fast else 100
    RAYS_ITERS  = 15  if args.fast else 60
    RAYS_SEARCH = 10  if args.fast else 15

    print(f"\n{'═'*60}")
    print(f"  BLACK-BOX — {DS.upper()} | eps {EPS} | seuil {THRESHOLD}")
    print(f"  Victimes : {args.victims}"
          + (f" (MLP={args.mlp_defense})" if args.victims == "defended" else ""))
    print(f"  Familles : Score={RUN_SCORE} Transfer={RUN_TRANSFER} Decision={RUN_DECISION}")
    print(f"  seeds {N_RUNS} | device {DEVICE} | fast {args.fast}")
    print(f"{'═'*60}")

    # ── Chargement victimes ──
    if args.victims == "defended":
        X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims_defended(
            SAVE, DEVICE, args.mlp_defense)
    else:
        X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims(SAVE, DEVICE)
    victims = [("MLP", mlp_w), ("LogReg", logreg_w), ("XGBoost", xgb_w)]

    # Données pour les substituts (toujours clean, même en mode defended)
    X_train = np.load(SAVE / "X_train.npy")
    y_train = np.load(SAVE / "y_train.npy")
    X_val   = np.load(SAVE / "X_val.npy")
    y_val   = np.load(SAVE / "y_val.npy")
    X_train, y_train = subsample_for_substitute(X_train, y_train)

    timestamps_test, has_ts_file = load_timestamps(SAVE)
    if args.timestamps == "on" and not has_ts_file:
        raise SystemExit("--timestamps on : timestamps_test.npy absent.")
    do_ts = has_ts_file and args.timestamps != "off"

    # ── Reprise depuis checkpoints par famille ──
    families = [f for f, run in [("score", RUN_SCORE), ("transfer", RUN_TRANSFER),
                                 ("decision", RUN_DECISION)] if run]
    results   = {f: [] for f in families}
    persample = {f: [] for f in families}
    for f in families:
        tmp = RES / f"blackbox_{f}_{TAG}_tmp.csv"
        if tmp.exists():
            results[f] = pd.read_csv(tmp).to_dict("records")
            print(f"  Reprise {f} : {len(results[f])} résultats déjà présents")

    def existing(f):
        return pd.DataFrame(results[f]) if results[f] else None

    def checkpoint(f):
        pd.DataFrame(results[f]).to_csv(RES / f"blackbox_{f}_{TAG}_tmp.csv", index=False)

    t0 = time.time()

    for seed in SEEDS:
        print(f"\n{'═'*60}\n  SEED {seed+1}/{N_RUNS}  [{timedelta(seconds=int(time.time()-t0))}]\n{'═'*60}")

        # ─────────────── SCORE-BASED ───────────────
        if RUN_SCORE:
            print("  [Score-based]")
            for vic_name, vic_w in victims:
                X_eval, y_eval, X_atk, y_atk, idx_ev = build_per_model_eval(
                    X_test, y_test, vic_w, seed, EVAL_ATK, EVAL_NRM, DS)
                ts_eval = timestamps_test[idx_ev] if do_ts else None

                for atk_name, fn in [("Square", square_attack), ("NES", nes_attack)]:
                    if already_done(existing("score"), seed, atk_name, vic_name):
                        continue
                    set_all_seeds(seed)
                    X_adv = fn(vic_w, X_atk, y_atk, EPS)
                    r = eval_attack(vic_w, X_eval, y_eval, X_adv, atk_name, vic_name,
                                    threshold=THRESHOLD)
                    r.update({"seed": seed, "family": "Score-based", "eps": EPS,
                              "dataset": DS, "n_atk": int((y_eval == 1).sum())})
                    results["score"].append(r)
                    print(f"    {vic_name}/{atk_name}: ASR {r['asr']*100:.1f}%")
                    if do_ts:
                        add_persample(persample["score"], vic_w, X_eval, y_eval, X_adv,
                                      atk_name, vic_name, ts_eval, PN, seed, EPS, DS)
            checkpoint("score")

        # ─────────────── TRANSFER ───────────────
        if RUN_TRANSFER:
            print("  [Transfer] entraînement des substituts...")
            set_all_seeds(seed)
            sub1 = train_substitute(MLP,      X_train, y_train, X_val, y_val, DEVICE, f"Sub1-seed{seed}")
            sub2 = train_substitute(SmallMLP, X_train, y_train, X_val, y_val, DEVICE, f"Sub2-seed{seed}")
            sub3 = train_substitute(DeepMLP,  X_train, y_train, X_val, y_val, DEVICE, f"Sub3-seed{seed}")
            subs = [("Sub1-MLP", sub1), ("Sub2-SmallMLP", sub2), ("Sub3-DeepMLP", sub3)]

            eval_sets = {}
            for vic_name, vic_w in victims:
                eval_sets[vic_name] = build_per_model_eval(
                    X_test, y_test, vic_w, seed, EVAL_ATK, EVAL_NRM, DS)

            def record_transfer(full_attack, vic_name, vic_w, X_adv, sub_label):
                X_eval, y_eval, _, _, idx_ev = eval_sets[vic_name]
                rt = eval_transfer(X_eval, y_eval, X_adv, vic_w, sub_label, vic_name, full_attack)
                results["transfer"].append({
                    "attack": full_attack, "model": vic_name, "asr": rt["asr"],
                    "f1_clean": rt["f1_clean"], "f1_adv": rt["f1_adv"],
                    "rec_adv": rt["rec_adv"], "linf": rt["linf"],
                    "substitute": sub_label, "seed": seed, "family": "Transfer",
                    "eps": EPS, "dataset": DS, "n_atk": int((y_eval == 1).sum())})
                if do_ts:
                    ts_eval = timestamps_test[idx_ev]
                    add_persample(persample["transfer"], vic_w, X_eval, y_eval, X_adv,
                                  full_attack, vic_name, ts_eval, PN, seed, EPS, DS)

            # MI-FGSM / VMI-FGSM par substitut
            for sub_name, sub_w in subs:
                for vic_name, vic_w in victims:
                    _, _, X_atk_vic, y_atk_vic, _ = eval_sets[vic_name]
                    set_all_seeds(seed)
                    X_mi  = mi_fgsm(sub_w,  X_atk_vic, y_atk_vic, eps=EPS)
                    X_vmi = vmi_fgsm(sub_w, X_atk_vic, y_atk_vic, eps=EPS)
                    for atk_name, X_adv in [("MI-FGSM", X_mi), ("VMI-FGSM", X_vmi)]:
                        full = f"{atk_name}_{sub_name}"
                        if already_done(existing("transfer"), seed, full, vic_name):
                            continue
                        record_transfer(full, vic_name, vic_w, X_adv, sub_name)

            # Ensemble-MI (3 substituts)
            for vic_name, vic_w in victims:
                if already_done(existing("transfer"), seed, "Ensemble-MI", vic_name):
                    continue
                _, _, X_atk_vic, y_atk_vic, _ = eval_sets[vic_name]
                set_all_seeds(seed)
                X_ens = ensemble_mi_fgsm([sub1, sub2, sub3], X_atk_vic, y_atk_vic,
                                         eps=EPS, weights=[1/3, 1/3, 1/3])
                record_transfer("Ensemble-MI", vic_name, vic_w, X_ens, "Ensemble(S1+S2+S3)")
            checkpoint("transfer")

        # ─────────────── DECISION-BASED ───────────────
        if RUN_DECISION:
            print("  [Decision-based]")
            for vic_name, vic_w in victims:
                X_eval, y_eval, X_atk, y_atk, idx_ev = build_per_model_eval(
                    X_test, y_test, vic_w, seed, EVAL_ATK, EVAL_NRM, DS)
                X_atk_bb, y_atk_bb, sub_idx = subsample_bb(X_atk, y_atk, seed, MAX_DB)
                X_bb, y_bb = build_bb_eval(X_eval, y_eval, X_atk_bb)

                ts_bb = None
                if do_ts:
                    ts_all = timestamps_test[idx_ev]
                    ts_bb  = np.concatenate([ts_all[y_eval == 0], ts_all[y_eval == 1][sub_idx]])

                for atk_name, fn, kw in [
                    ("HSJA", hsja, {"iters": HSJA_ITERS, "n_est": HSJA_NEST}),
                    ("RayS", rays, {"iters": RAYS_ITERS, "search_steps": RAYS_SEARCH}),
                ]:
                    if already_done(existing("decision"), seed, atk_name, vic_name):
                        continue
                    print(f"    {vic_name}/{atk_name} sur {len(X_atk_bb)} attaques...")
                    set_all_seeds(seed)
                    X_adv_bb = fn(vic_w, X_atk_bb, y_atk_bb, EPS, **kw)
                    r = eval_attack(vic_w, X_bb, y_bb, X_adv_bb, atk_name, vic_name,
                                    threshold=THRESHOLD)
                    r.update({"seed": seed, "family": "Decision-based", "eps": EPS,
                              "dataset": DS, "n_atk": len(X_atk_bb)})
                    results["decision"].append(r)
                    print(f"      ASR {r['asr']*100:.1f}%")
                    if do_ts:
                        add_persample(persample["decision"], vic_w, X_bb, y_bb, X_adv_bb,
                                      atk_name, vic_name, ts_bb, PN, seed, EPS, DS)
                    checkpoint("decision")

        for f in families:
            checkpoint(f)

    # ── Écriture finale + nettoyage des tmp ──
    for f in families:
        write_family(f, results[f], persample[f], RES, TAG, do_ts)
        tmp = RES / f"blackbox_{f}_{TAG}_tmp.csv"
        if tmp.exists():
            tmp.unlink()
    print(f"\n✓ Terminé en {timedelta(seconds=int(time.time()-t0))}")


if __name__ == "__main__":
    main()