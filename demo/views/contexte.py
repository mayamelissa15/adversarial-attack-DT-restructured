import streamlit as st

from lib import theme
from lib.data import DATASET_META, feature_names

ds = st.session_state.get("dataset", "swat")

theme.banner("Données")

cols = st.columns(2)
for col, key in zip(cols, ["swat", "batadal"]):
    d = DATASET_META[key]
    feats = feature_names(key)
    with col:
        st.markdown(f'''
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <div style="font-size:19px;font-weight:800;">{d["name"]}</div>
            {theme.badge("actif", "final") if key == ds else ""}
          </div>
          <div style="font-size:12.5px;color:{theme.INK_SOFT};margin:4px 0 14px;">{d["origin"]}</div>
        </div>''', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            theme.stat_tile("Capteurs", str(len(feats)) if feats else "-")
        with c2:
            if d["stages"]:
                theme.stat_tile("Sous-systèmes", str(len(d["stages"])))
            else:
                n_tanks = len([f for f in feats if f.startswith("L_T")])
                theme.stat_tile("Réservoirs", str(n_tanks) if n_tanks else "-")

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)
feats = feature_names(ds)
if feats:
    st.dataframe(
        {"#": list(range(1, len(feats) + 1)), "tag": feats},
        hide_index=True, use_container_width=True, height=260,
    )
else:
    theme.note("feature_names.json introuvable.")
