# Guide du code : these_adv

## Arborescence

```
these_adv/
├── data/raw/
│   ├── swat/merged.csv
│   └── batadal/BATADAL_dataset03.csv, BATADAL_dataset04.csv
│
├── src/
│   ├── 00-train.py
│   ├── common.py
│   ├── common_whitebox.py
│   ├── models.py
│   ├── whitebox.py
│   ├── whitebox_run.py
│   ├── blackbox.py
│   ├── blackbox_run.py
│   ├── transfer.py
│   ├── defenses.py
│   ├── evaluate.py
│   ├── evaluate_blackbox.py
│   └── plot_results.py
│
├── artifacts/<dataset>/     -> modèles entraînés + splits de données (généré par le code)
├── results/<dataset>/       -> métriques et figures (généré par le code)
│   └── plots/
│
└── demo/                    -> interface Streamlit pour la soutenance
    ├── app.py
    ├── lib/data.py
    ├── lib/theme.py
    └── views/contexte.py, modeles.py, defenses.py, demonstration.py
```

Deux dossiers ne contiennent pas de code écrit à la main : `artifacts/` (modèles) et `results/` (résultats). Tout y est régénéré en relançant les scripts de `src/`.

---

## src/common.py

Fonctions partagées par tout le reste du code : chargement des datasets, calcul de l'ASR (taux de succès des attaques), constantes globales (`SEED=42`, `THRESHOLD=0.5`).

**Résultat produit :** aucun (fichier de fonctions, pas de script exécutable).

## src/common_whitebox.py

Fonctions partagées par les scripts d'attaque boîte blanche : construit le même jeu d'évaluation pour FGSM/PGD/C&W, pour que les ASR soient comparables entre attaques.

**Résultat produit :** aucun.

## src/models.py

Définit les architectures (MLP) et les "wrappers" qui donnent la même interface (predict, gradient) à MLP, LogReg et XGBoost. XGBoost n'a pas de gradient exact (c'est un ensemble d'arbres), donc son gradient est approximé par différences finies.

**Résultat produit :** aucun.

---

## src/00-train.py

Entraîne les 3 modèles de référence (MLP, LogReg, XGBoost) sur un dataset (SWaT ou BATADAL), avec un split train/validation/test.

**Résultat produit :**
- `artifacts/<dataset>/` : les modèles entraînés (`best_mlp.pt`, `logreg.pkl`, `xgb.json`), le scaler, les splits de données
- `results/<dataset>/train_results.json` : accuracy, F1, précision, rappel de chaque modèle en clair (sans attaque)

---

## src/whitebox.py

Implémente les 3 attaques boîte blanche : FGSM, PGD, C&W. L'attaquant a un accès complet au modèle.

**Résultat produit :** aucun (bibliothèque de fonctions, appelée par whitebox_run.py).

## src/whitebox_run.py

Lance les attaques de whitebox.py sur les 3 modèles, sur 10 graines aléatoires, et sauvegarde les résultats.

Commande : `python src/whitebox_run.py --dataset swat --eps 0.1 --attack all`

**Résultat produit :**
- `results/<dataset>/whitebox_{fgsm,pgd,cw}_<dataset>_eps0.1.csv` et `.json` : ASR médiane par modèle et par attaque
- `results/<dataset>/whitebox_persample_*.csv` : résultat attaque par attaque, échantillon par échantillon

---

## src/blackbox.py

Implémente 4 attaques boîte noire : Square et NES (accès aux probabilités du modèle), HSJA et RayS (accès au label prédit seulement, pas aux probabilités).

**Résultat produit :** aucun (bibliothèque, appelée par blackbox_run.py).

## src/transfer.py

Implémente les attaques par transfert : on entraîne des modèles "substituts", on attaque ces substituts, et on teste si l'attaque marche aussi sur le vrai modèle, sans jamais le requêter.

**Résultat produit :** aucun (appelé par blackbox_run.py).

## src/blackbox_run.py

Lance les 3 familles d'attaque boîte noire (score, décision, transfert) sur les modèles, avec reprise possible si le script est interrompu.

Commande : `python src/blackbox_run.py --dataset swat --eps 0.1`

**Résultat produit :**
- `results/<dataset>/blackbox_{score,decision,transfer}_<dataset>_eps0.1.csv` et `.json`
- avec `--victims defended` : mêmes fichiers avec suffixe `_defended` (attaque des modèles durcis)

---

## src/defenses.py

Entraîne 7 versions durcies des modèles : entraînement adversarial (MLP) et augmentation par données adversariales (MLP, LogReg, XGBoost), contre les attaques FGSM, PGD et Square.

Commande : `python src/defenses.py --dataset swat`

**Résultat produit :** `artifacts/<dataset>/` : les modèles durcis (`mlp_at_fgsm.pt`, `mlp_at_pgd.pt`, `mlp_at_square.pt`, `logreg_aug_fgsm.pkl`, `logreg_aug_square.pkl`, `xgb_iter_fgsm_r3.json`, `xgb_iter_square_r3.json`)

---

## src/evaluate.py

Réattaque directement (en boîte blanche) les modèles durcis, pour mesurer si la défense marche vraiment (et pas juste contre des attaques précalculées).

Commande : `python src/evaluate.py --dataset swat --eps 0.1`

**Résultat produit :** `results/<dataset>/defense_results.json` : ASR après défense, comparée à l'ASR de base.

## src/evaluate_blackbox.py

Même chose mais en boîte noire, sur les 7 défenses, pour tester si une défense entraînée contre une attaque marche aussi contre les autres.

Commande : `python src/evaluate_blackbox.py --dataset swat --eps 0.1`

**Résultat produit :** `results/<dataset>/defense_results_blackbox.json`

---

## src/plot_results.py

Génère les graphiques (barres, boîtes à moustaches) à partir des fichiers csv/json déjà produits. Ne recalcule rien.

Commande : `python src/plot_results.py --dataset swat --eps 0.1`

**Résultat produit :** `results/<dataset>/plots/*.png`

---

## demo/ (interface Streamlit pour la soutenance)

| Fichier | Rôle |
|---|---|
| `app.py` | Point d'entrée, sélecteur de dataset, navigation entre pages |
| `lib/data.py` | Lit les fichiers de `results/` et `artifacts/`, sans rien recalculer |
| `lib/theme.py` | Couleurs et composants visuels |
| `views/contexte.py` | Page "Données" : métadonnées SWaT/BATADAL |
| `views/modeles.py` | Page "Modèles" : performance en clair des 3 modèles |
| `views/defenses.py` | Page "Défenses" : les 7 défenses et leur ASR après attaque |
| `views/demonstration.py` | Page "Attaques" : anime les résultats déjà calculés (aucun calcul en direct) |

Lancement : `streamlit run demo/app.py`

**Résultat produit :** aucun fichier, c'est une interface de lecture.

---

## Où trouver les résultats chiffrés

- Performance en clair : `results/<dataset>/train_results.json`
- ASR par attaque : `results/<dataset>/whitebox_*.json` et `blackbox_*.json`
- ASR après défense : `results/<dataset>/defense_results.json` et `defense_results_blackbox.json`
- Figures : `results/<dataset>/plots/`

État au 2026-07-17 : seul eps=0.1 a des résultats d'attaque. SWaT n'a pas encore la matrice complète des 7 défenses en boîte noire (une seule défense par défaut par modèle). BATADAL, si.
