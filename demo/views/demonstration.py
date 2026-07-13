import time

import numpy as np
import pandas as pd
import streamlit as st

from lib import theme
from lib.data import (
    BBD_ATTACKS, BBS_ATTACKS, DATASET_META, MODELS, blackbox_decision_asr,
    blackbox_score_asr, feature_names, load_test_arrays, persample_blackbox_decision,
    persample_blackbox_score, persample_whitebox, transfer_asr, whitebox_asr,
)

ANIMATION_SECONDS = 3.5

ds = st.session_state.get("dataset", "swat")
meta = DATASET_META[ds]

theme.banner("Attaques")

if meta["stages"]:
    cols = st.columns(len(meta["stages"]))
    for c, s in zip(cols, meta["stages"]):
        with c:
            st.markdown(f'''
            <div class="stage">
              <div class="sid">{s["id"]}</div>
              <div style="margin-top:3px;">{s["label"]}</div>
            </div>''', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

left, right = st.columns([1, 1.3], gap="large")

with left:
    threat = st.segmented_control("Menace", ["White-box", "Black-box"], default="White-box", key="demo_threat")

    attack, variant, family = None, None, None

    if threat == "White-box":
        wb = whitebox_asr(ds)
        options = list(wb.keys())
        if options:
            attack = st.segmented_control("Attaque", options, default=options[0], key="demo_wb_attack")
    else:
        family = st.segmented_control(
            "Famille", ["Score-based", "Decision-based", "Transfert"],
            default="Score-based", key="demo_bb_family",
        )
        if family == "Score-based":
            options = BBS_ATTACKS if blackbox_score_asr(ds) else []
            if options:
                attack = st.segmented_control("Attaque", options, default=options[0], key="demo_bbs_attack")
        elif family == "Decision-based":
            options = BBD_ATTACKS if blackbox_decision_asr(ds) else []
            if options:
                attack = st.segmented_control("Attaque", options, default=options[0], key="demo_bbd_attack")
        else:
            tr = transfer_asr(ds)
            options = list(tr.get("MLP", {}).keys()) if tr else []
            if options:
                variant = st.selectbox("Variante", options, key="demo_tr_variant")

    model = st.segmented_control("Modèle cible", MODELS, default="MLP", key="demo_model")

    persample_df = None
    if threat == "White-box" and attack:
        persample_df = persample_whitebox(ds, attack)
    elif threat == "Black-box" and family == "Score-based" and attack:
        persample_df = persample_blackbox_score(ds)
    elif threat == "Black-box" and family == "Decision-based" and attack:
        persample_df = persample_blackbox_decision(ds)

    full_pool = None
    if persample_df is not None and ds == "swat" and model:
        mask = persample_df["model"] == model
        if attack and "attack" in persample_df.columns:
            mask &= persample_df["attack"] == attack
        full_pool = persample_df[mask]

    launch_key = (ds, threat, family, attack, model, variant)
    launched = st.session_state.setdefault("launched", {})

    if full_pool is not None and len(full_pool):
        n_total = len(full_pool)
        if st.button(f"Lancer l'attaque sur {n_total} échantillons", type="primary", key="demo_launch"):
            run = full_pool.sample(frac=1, random_state=0).reset_index(drop=True)["success"].tolist()
            prog = st.progress(0.0)
            status = st.empty()
            sleep_s = max(ANIMATION_SECONDS / n_total, 0.002)
            seen = []
            for i, flag in enumerate(run, start=1):
                seen.append(flag)
                prog.progress(i / n_total)
                status.markdown(f"**{sum(seen)}** évadés / **{i - sum(seen)}** détectés  ({i}/{n_total})")
                time.sleep(sleep_s)
            prog.empty()
            status.empty()
            launched[launch_key] = seen

with right:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)

    seen = launched.get(launch_key)
    if seen:
        asr = 100 * sum(seen) / len(seen)
        robust = 100 - asr
        c1, c2 = st.columns(2)
        with c1:
            theme.stat_tile("Évasion (ASR)", f"{asr:.1f}%")
        with c2:
            theme.stat_tile("Détection maintenue", f"{robust:.1f}%")
        theme.asr_curve_chart(pd.Series(seen), len(seen) - 1, theme.MODEL_COLOR[model])
    elif threat == "Black-box" and family == "Transfert":
        tr = transfer_asr(ds)
        if tr and variant:
            row = {m: tr.get(m, {}).get(variant, {}).get("evasion_rate_median") for m in MODELS}
            theme.asr_bar_chart(row)
        else:
            theme.note("Pas encore calculé pour ce jeu de données.")
    elif full_pool is not None and len(full_pool):
        theme.note("Clique sur Lancer l'attaque pour voir le résultat réel.")
    else:
        theme.note("Pas encore calculé pour ce jeu de données.")

    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data
def attack_window(dataset: str, pad: int = 150):
    X, y, ts = load_test_arrays(dataset)
    feats = feature_names(dataset)
    idx = np.where(y == 1)[0]
    if len(idx) == 0:
        return None
    edges = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, len(idx) - 1]
    block = int(np.argmax(ends - starts))
    lo, hi = idx[starts[block]], idx[ends[block]]
    a, b = max(0, lo - pad), min(len(y), hi + pad)

    prefer = ["LIT101", "FIT101"] if dataset == "swat" else feats[:2]
    cols = [f for f in prefer if f in feats] or feats[:2]
    col_idx = [feats.index(c) for c in cols]

    df = pd.DataFrame(X[a:b][:, col_idx], columns=cols)
    df["temps"] = pd.to_datetime(ts[a:b])
    return df.set_index("temps"), cols


win = attack_window(ds)
if win is not None:
    df, cols = win
    st.line_chart(df[cols], height=180)
