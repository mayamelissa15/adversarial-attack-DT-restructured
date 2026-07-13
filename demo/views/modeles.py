import pandas as pd
import streamlit as st

from lib import theme
from lib.data import MODELS, train_results

ds = st.session_state.get("dataset", "swat")

theme.banner("Modèles")

clean = train_results(ds)
if not clean:
    theme.note("train_results.json introuvable.")
else:
    rows = []
    for m in MODELS:
        c = clean.get(m)
        if not c:
            continue
        rows.append({
            "Modèle": m, "Accuracy (%)": c["clean_accuracy"], "F1": c["f1"],
            "Précision": c["precision"], "Rappel": c["recall"],
        })
    df = pd.DataFrame(rows).set_index("Modèle")
    st.dataframe(
        df.style.format({"Accuracy (%)": "{:.2f}", "F1": "{:.4f}", "Précision": "{:.4f}", "Rappel": "{:.4f}"})
                 .background_gradient(cmap="Purples", subset=["F1"]),
        use_container_width=True,
    )
    st.caption(
        "F1 quasi parfait pour XGBoost, sans lien avec sa robustesse sous attaque."
        if ds == "swat" else
        "MLP peine en clean malgré un rappel correct, classes déséquilibrées."
    )
