"""
demo/app.py : point d'entrée du support de soutenance interactif.

Lance avec :
    .venv/bin/streamlit run demo/app.py

La page (soutenance.html) est autonome (fonts, données réelles eps=0.1,
JS embarqués) : Streamlit sert uniquement de conteneur pour pouvoir la
lancer avec `streamlit run` comme l'ancienne démo multi-pages, qu'elle
remplace. Pour mettre à jour les chiffres, régénérer soutenance.html
(voir la conversation Claude qui l'a produit) plutôt qu'éditer ce fichier.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Robustesse Adversariale — Console de Soutenance",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    "<style>.block-container{padding:0 !important;max-width:100% !important;} "
    "header[data-testid='stHeader']{height:0;} iframe{display:block;}</style>",
    unsafe_allow_html=True,
)

html_path = Path(__file__).resolve().parent / "soutenance.html"
components.html(html_path.read_text(encoding="utf-8"), height=2400, scrolling=True)
