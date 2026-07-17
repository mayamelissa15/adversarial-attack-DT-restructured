# these_adv

Code de thèse : robustesse adversariale de modèles de détection d'anomalies/intrusions (ML) pour les systèmes de contrôle industriel (ICS).

Deux jeux de données :
- **SWaT**   testbed réel de traitement d'eau (iTrust, Singapour), 51 capteurs/actionneurs, 6 étapes de process (P1–P6).
- **BATADAL**   réseau de distribution d'eau simulé (challenge EPANET), 43 features.

Trois modèles de référence (MLP, LogReg, XGBoost) sont entraînés en détection propre, puis attaqués sous 4 familles de menaces :
- **Boîte blanche** (accès au gradient) : FGSM, PGD, C&W
- **Boîte noire à score** (accès aux probabilités) : Square, NES
- **Boîte noire à décision** (accès au label seul) : HSJA, RayS
- **Par transfert** (aucun accès au modèle cible) : MI-FGSM, VMI-FGSM via modèles substituts

Sept versions durcies des modèles (entraînement adversarial, augmentation par données adversariales) sont ensuite réévaluées contre ces mêmes attaques pour mesurer l'efficacité des défenses.

## Installation

Deux environnements séparés :

**Entraînement / attaques / défenses** (torch, xgboost, scikit-learn   plus lourd) :
```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install torch numpy pandas scikit-learn xgboost matplotlib
```

**Démo Streamlit** (lecture seule, aucune dépendance ML) :
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements.txt
```

Les données brutes (`data/raw/swat/merged.csv`, `data/raw/batadal/*.csv`) ne sont pas versionnées (voir `.gitignore`) et doivent être placées manuellement dans `data/raw/`.

## Utilisation

Pipeline complet pour un dataset (`swat` ou `batadal`) :

```bash
# 1. Entraîner les 3 modèles de référence
python src/00-train.py --dataset swat

# 2. Attaques boîte blanche
python src/whitebox_run.py --dataset swat --eps 0.1 --attack all

# 3. Attaques boîte noire (score, décision, transfert)
python src/blackbox_run.py --dataset swat --eps 0.1

# 4. Entraîner les 7 défenses
python src/defenses.py --dataset swat

# 5. Réévaluer les défenses (boîte blanche / boîte noire)
python src/evaluate.py --dataset swat --eps 0.1
python src/evaluate_blackbox.py --dataset swat --eps 0.1

# 6. Générer les figures
python src/plot_results.py --dataset swat --eps 0.1
```

Chaque étape lit les artefacts produits par la précédente (`artifacts/<dataset>/`) et écrit ses résultats dans `results/<dataset>/`.

## Démo

Interface Streamlit de lecture (aucun calcul en direct, uniquement les résultats déjà produits) :

```bash
streamlit run demo/app.py
```

## Structure

```
these_adv/
├── data/raw/          données brutes (non versionnées)
├── src/               pipeline entraînement → attaques → défenses → évaluation
├── artifacts/<dataset>/  modèles entraînés (régénéré par src/)
├── results/<dataset>/    métriques et figures (régénéré par src/)
└── demo/               interface Streamlit pour la soutenance
```

Voir [GUIDE_CODE.md](GUIDE_CODE.md) pour le détail de chaque script (entrées, sorties, commandes) et l'emplacement des résultats chiffrés.
