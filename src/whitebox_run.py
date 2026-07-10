"""
whitebox_run.py
═══════════════

Usage :
  python src/whitebox_run.py --dataset swat    --eps 0.1 --attack all
  python src/whitebox_run.py --dataset swat    --eps 0.1 --attack fgsm
  python src/whitebox_run.py --dataset batadal --eps 0.3 --attack pgd --fast

Parallélisation : lancer plusieurs process avec des --attack différents donne
EXACTEMENT les mêmes résultats que --attack all (eval set via RNG local seedé +
re-seed global avant chaque attaque → ordre-indépendant).

Sorties (par attaque, mêmes noms qu'avant → merge inchangé) :
  results/<ds>/whitebox_<attack>_<ds>_eps<eps>.csv / .json
  results/<ds>/whitebox_<attack>_<ds>_eps<eps>_tmp.csv            (checkpoint)
  results/<ds>/whitebox_persample_<attack>_<ds>_eps<eps>.csv      (timestamps)
"""

import json
import warnings

import pandas as pd
warnings.filterwarnings("ignore")

from common import THRESHOLD, eval_attack, eval_attack_persample
from common_whitebox import (
    build_arg_parser, setup_paths, get_device, eval_sizes,
    load_victims, build_per_model_eval, load_timestamps, set_all_seeds,
    VICTIMS_SPEC,
)
from whitebox import (
    fgsm_mlp, fgsm_logreg, fgsm_xgb,
    pgd_mlp,  pgd_logreg,  pgd_xgb,
    cw_mlp,   cw_logreg,   cw_xgb,
    PGD_ALPHA_K,
)

# Étiquette lisible + fragment de nom de fichier, par attaque.
ATTACK_LABEL = {"fgsm": "FGSM", "pgd": "PGD", "cw": "C&W"}
ATTACK_FILE  = {"fgsm": "fgsm", "pgd": "pgd", "cw": "cw"}


# ══════════════════════════════════════════════════════════════
# DISPATCH D'ATTAQUE (choisit la bonne fonction selon le modèle)
# ══════════════════════════════════════════════════════════════

def run_one_attack(attack, is_lr, is_xgb, vic_w, X_atk, y_atk, eps,
                   pgd_iters, pgd_restarts, cw_iters):
    if attack == "fgsm":
        fn = fgsm_logreg if is_lr else (fgsm_xgb if is_xgb else fgsm_mlp)
        return fn(vic_w, X_atk, y_atk, eps)

    if attack == "pgd":
        fn    = pgd_logreg if is_lr else (pgd_xgb if is_xgb else pgd_mlp)
        alpha = eps / PGD_ALPHA_K
        return fn(vic_w, X_atk, y_atk, eps,
                  iters=pgd_iters, restarts=pgd_restarts, alpha=alpha)

    if attack == "cw":
        fn = cw_logreg if is_lr else (cw_xgb if is_xgb else cw_mlp)
        return fn(vic_w, X_atk, y_atk, eps, iters=cw_iters)

    raise ValueError(f"attaque inconnue : {attack}")


# ══════════════════════════════════════════════════════════════
# ÉCRITURE DES SORTIES (par attaque — compatible merge existant)
# ══════════════════════════════════════════════════════════════

def write_outputs(attack, rows, persample_rows, res_dir, tag, has_ts):
    afile = ATTACK_FILE[attack]
    label = ATTACK_LABEL[attack]

    df = pd.DataFrame(rows)
    csv_path = res_dir / f"whitebox_{afile}_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV → {csv_path}")

    summary = (df.groupby(["attack", "model"])["asr"]
                 .agg(["median", "std", "min", "max"]).round(4))
    print(f"\n{'═'*55}\n  RÉSUMÉ {label} — {tag}\n{'═'*55}")
    print(summary.to_string())

    out = {}
    for model_name in df["model"].unique():
        vals = df[df["model"] == model_name]["asr"]
        out[model_name] = {label: {
            "evasion_rate_median": round(float(vals.median()) * 100, 2),
            "evasion_rate_std":    round(float(vals.std())    * 100, 2),
            "evasion_rate_min":    round(float(vals.min())    * 100, 2),
            "evasion_rate_max":    round(float(vals.max())    * 100, 2),
        }}
    json_path = res_dir / f"whitebox_{afile}_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ JSON → {json_path}")

    if has_ts and persample_rows:
        dfp = pd.DataFrame(persample_rows)
        p_path = res_dir / f"whitebox_persample_{afile}_{tag}.csv"
        dfp.to_csv(p_path, index=False)
        print(f"✓ CSV par-échantillon → {p_path}  ({len(dfp)} lignes)")


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    args = build_arg_parser("Runner white-box unifié (FGSM/PGD/C&W)").parse_args()
    DATASET, EPS, N_RUNS, FAST = args.dataset, args.eps, args.n_runs, args.fast
    PERSAMPLE_N = args.persample_n

    attacks = ["fgsm", "pgd", "cw"] if args.attack == "all" else [args.attack]

    SAVE_DIR, RES_DIR, TAG = setup_paths(DATASET, EPS)
    DEVICE = get_device()
    SEEDS  = list(range(N_RUNS))
    EVAL_ATK_SIZE, EVAL_NRM_SIZE = eval_sizes(DATASET)

    PGD_ITERS    = 50  if FAST else 200
    PGD_RESTARTS = 3   if FAST else 10
    CW_ITERS     = 150 if FAST else 500

    print(f"\n{'═'*55}")
    print(f"  Attaque(s) : {', '.join(ATTACK_LABEL[a] for a in attacks)}")
    print(f"  Dataset    : {DATASET.upper()}")
    print(f"  Epsilon    : {EPS}")
    print(f"  Seuil      : {THRESHOLD}")
    print(f"  N_RUNS     : {N_RUNS}")
    print(f"  Device     : {DEVICE}")
    print(f"  FAST       : {FAST}")
    if "pgd" in attacks:
        print(f"  PGD        : {PGD_ITERS} iters × {PGD_RESTARTS} restarts")
    if "cw" in attacks:
        print(f"  C&W        : {CW_ITERS} iters")
    print(f"  Sorties    : {RES_DIR}")
    print(f"{'═'*55}")

    X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims(SAVE_DIR, DEVICE)
    victims = {"MLP": mlp_w, "LogReg": logreg_w, "XGBoost": xgb_w}
    timestamps_test, has_ts = load_timestamps(SAVE_DIR)

    # Un accumulateur par attaque
    results   = {a: [] for a in attacks}
    persample = {a: [] for a in attacks}

    for seed in SEEDS:
        print(f"\n{'═'*55}")
        print(f"  SEED {seed+1}/{N_RUNS}  —  {DATASET.upper()}  eps={EPS}")
        print(f"{'═'*55}")

        for vic_name, is_lr, is_xgb in VICTIMS_SPEC:
            vic_w = victims[vic_name]

            # Eval set construit UNE fois par (seed, modèle) — RNG local, donc
            # identique pour les 3 attaques (comparabilité garantie).
            X_eval, y_eval, X_atk, y_atk, idx_ev = build_per_model_eval(
                X_test, y_test, vic_w, seed, EVAL_ATK_SIZE, EVAL_NRM_SIZE, DATASET)

            print(f"\n  ── {vic_name} {'─'*(45-len(vic_name))}")
            print(f"  Eval set : {len(X_eval)} exemples "
                  f"({int((y_eval==1).sum())} attaques, {int((y_eval==0).sum())} normaux)")

            for attack in attacks:
                # Re-seed AVANT chaque attaque → résultats ordre-indépendants,
                # donc --attack all == runs séparés bit-à-bit.
                set_all_seeds(seed)
                label = ATTACK_LABEL[attack]
                print(f"  [{label}]")

                X_adv = run_one_attack(attack, is_lr, is_xgb, vic_w,
                                       X_atk, y_atk, EPS,
                                       PGD_ITERS, PGD_RESTARTS, CW_ITERS)

                r = eval_attack(vic_w, X_eval, y_eval, X_adv,
                                label, vic_name, threshold=THRESHOLD)
                r.update({"seed": seed, "family": "Whitebox",
                          "eps": EPS, "dataset": DATASET,
                          "n_atk": int((y_eval == 1).sum())})
                results[attack].append(r)
                print(f"    ASR={r['asr']*100:.1f}%  "
                      f"F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}  "
                      f"margin_adv_mean={r.get('margin_adv_mean', float('nan')):.4f}  "
                      f"L∞mean={r.get('linf_mean', float('nan')):.4f}")

                if has_ts:
                    ts_eval = timestamps_test[idx_ev]
                    recs = eval_attack_persample(
                        vic_w, X_eval, y_eval, X_adv, label, vic_name,
                        timestamps_full=ts_eval, threshold=THRESHOLD,
                        max_samples=PERSAMPLE_N, seed=seed)
                    for rec in recs:
                        rec.update({"eps": EPS, "dataset": DATASET})
                    persample[attack].extend(recs)

        # Checkpoint après chaque seed (par attaque)
        for attack in attacks:
            afile = ATTACK_FILE[attack]
            pd.DataFrame(results[attack]).to_csv(
                RES_DIR / f"whitebox_{afile}_{TAG}_tmp.csv", index=False)
            if has_ts and persample[attack]:
                pd.DataFrame(persample[attack]).to_csv(
                    RES_DIR / f"whitebox_persample_{afile}_{TAG}_tmp.csv", index=False)
        print(f"\n  ✓ Checkpoint seed {seed} sauvegardé")

    # Écriture finale, une passe par attaque
    for attack in attacks:
        write_outputs(attack, results[attack], persample[attack],
                      RES_DIR, TAG, has_ts)


if __name__ == "__main__":
    main()