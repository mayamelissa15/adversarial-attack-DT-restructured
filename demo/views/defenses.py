import streamlit as st

from lib import theme
from lib.data import DEFENSES, MODEL_COLOR, defense_results, defense_trained

ds = st.session_state.get("dataset", "swat")

theme.banner("Défenses", "Sept modèles durcis, ASR post-défense mis à jour dès evaluate.py lancé")

results = defense_results(ds)

cols = st.columns(3)
for i, d in enumerate(DEFENSES):
    trained = defense_trained(ds, d["file"])
    res = results.get(d["model"], {}).get(d["name"]) if results else None
    with cols[i % 3]:
        st.markdown(f'''
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <span style="color:{MODEL_COLOR[d["model"]]};font-weight:700;font-size:11px;text-transform:uppercase;">{d["model"]}</span>
              <div style="font-weight:700;font-size:14px;">{d["name"]}</div>
            </div>
            {theme.badge(d["threat"], "pending")}
          </div>
          <div style="font-size:12px;color:{theme.INK_SOFT};margin:6px 0 10px;line-height:1.5;">
            {d["method"]} (ε_entraînement = {d["eps"]})
          </div>
          <div style="font-size:11.5px;">
            {theme.badge("modèle entraîné", "final") if trained else theme.badge("modèle absent", "pending")}
          </div>
        </div>''', unsafe_allow_html=True)

        if res:
            st.markdown("<br/>", unsafe_allow_html=True)
            for attack_label, e in res.items():
                delta = e.get("delta_asr")
                st.metric(
                    f"ASR sous {attack_label}",
                    f"{e['asr_median']:.1f}%",
                    delta=f"{delta:+.1f} pts vs baseline" if delta is not None else None,
                    delta_color="inverse",
                )
        else:
            st.caption("ASR post-défense : en attente (`evaluate.py` non lancé).")
