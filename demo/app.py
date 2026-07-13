"""
demo/app.py : point d'entrée du support de soutenance interactif.

Lance avec :
    streamlit run demo/app.py

Toutes les données affichées sont lues directement dans results/ et
artifacts/ (racine du projet) : rien n'est recalculé ici, et rien n'est
inventé, un panneau sans fichier correspondant affiche "en attente".
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import theme
from lib.data import DATASET_META

st.set_page_config(
    page_title="Démonstration Attaques Adversariales",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_css()

with st.sidebar:
    st.markdown("**Robustesse adversariale des IDS industriels**")
    st.segmented_control(
        "Jeu de données",
        options=["swat", "batadal"],
        format_func=lambda k: DATASET_META[k]["name"],
        default="swat",
        key="dataset",
    )
    st.divider()

pages = [
    st.Page("views/demonstration.py", title="Attaques", default=True),
    st.Page("views/contexte.py", title="Données"),
    st.Page("views/modeles.py", title="Modèles"),
    st.Page("views/defenses.py", title="Défenses"),
]

nav = st.navigation(pages)
nav.run()
