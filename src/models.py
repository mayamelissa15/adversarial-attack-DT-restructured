"""
models.py
═════════
Définitions des MODÈLES

Ce fichier contient DEUX choses :
  1. Les architectures PyTorch : MLP (cible) + SmallMLP / DeepMLP (substituts).
  2. Les "wrappers" : une couche fine qui donne à chaque modèle (MLP, LogReg,
     XGBoost) une interface IDENTIQUE (predict / predict_proba / logits_np /
     gradient), pour que les scripts d'attaque puissent les manipuler de la
     même façon sans se soucier de la bibliothèque sous-jacente.

#fichier independant peut etre importé 
"""

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════
# ARCHITECTURE CIBLE : MLP
# ══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Perceptron multicouche binaire — le modèle NEURONAL cible.
    Architecture : input → 128 → 64 → 32 → 1 (logit).

    La dernière couche renvoie un LOGIT (score réel non borné), pas une
    probabilité. La sigmoïde est appliquée séparément (dans la loss pendant
    l'entraînement, ou dans les wrappers pour l'inférence). C'est le schéma
    recommandé avec BCEWithLogitsLoss (plus stable numériquement).
    """
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# ARCHITECTURES SUBSTITUTS (pour les attaques black-box par transfert)
# ══════════════════════════════════════════════════════════════

class SmallMLP(nn.Module):
    """Substitut LÉGER : input → 64 → 32 → 1."""
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """Substitut EXPRESSIF (plus profond) : input → 256 → 128 → 64 → 32 → 1."""
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# WRAPPER MLP
# ══════════════════════════════════════════════════════════════

class MLPWrapper:
    """
    Enrobe un modèle PyTorch pour exposer l'interface commune aux attaques.

    Le seuil par défaut est 0.5 pour rester cohérent avec common.THRESHOLD ;
    en pratique les scripts passeront TOUJOURS common.THRESHOLD explicitement
    (source de vérité unique).
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device

    def predict(self, X, threshold=0.5):
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_t).squeeze(-1)
            proba  = torch.sigmoid(logits).cpu().numpy()
        return (proba >= threshold).astype(int)

    def predict_proba(self, X):
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_t).squeeze(-1)
            proba  = torch.sigmoid(logits).cpu().numpy()
        return proba

    def logits_np(self, X):
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(x_t).squeeze(-1).cpu().numpy()


# ══════════════════════════════════════════════════════════════
# WRAPPER LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════

class LogRegWrapper:
    """
    Enrobe une LogisticRegression scikit-learn.

    Modèle linéaire : logit(x) = w·x + b. Le gradient du logit par rapport à
    l'entrée x est donc SIMPLEMENT le vecteur de coefficients w — constant,
    exact, sans approximation.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X, threshold=0.5):
        return (self.model.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def logits_np(self, X):
        return X @ self.model.coef_[0] + self.model.intercept_[0]

    def grad_bce(self, X, y):
        """
        Gradient EXACT de la BCE par rapport à x :
            ∇_x L_BCE = (p − y) · w
        Utilisé par FGSM / PGD.
        """
        p = 1 / (1 + np.exp(-self.logits_np(X)))          # (N,)
        return (p - y)[:, None] * self.model.coef_        # (N, D)

    def grad_logit(self, X):
        """
        Gradient EXACT du logit par rapport à x :
            ∂logit/∂x = w   (constant pour un modèle linéaire)
        Utilisé par C&W.
        """
        return np.tile(self.model.coef_, (len(X), 1))     # (N, D)


# ══════════════════════════════════════════════════════════════
# WRAPPER XGBOOST — gradient NUMÉRIQUE (différences finies centrées)
# ══════════════════════════════════════════════════════════════

class XGBoostWrapper:
    """
    Enrobe un XGBClassifier avec un gradient approché par différences finies.

    Pourquoi différences finies ?
    ─────────────────────────────
    XGBoost est un ensemble d'arbres. Un arbre est constant par morceaux : sa
    dérivée exacte est 0 presque partout et indéfinie aux nœuds de décision.
    La backpropagation est donc impossible. On APPROXIME le gradient du logit
    g = log(p/(1−p)) en évaluant le modèle deux fois par feature (±ε_fd) :

        ∂g/∂x_j ≈ [g(x + ε_fd·eⱼ) − g(x − ε_fd·eⱼ)] / (2·ε_fd)

    C'est la formule centrée d'ordre 2 : elle annule l'erreur de premier ordre,
    approximation en O(ε_fd²).

    Coût : 2·D forward passes par batch. Valeur recommandée : ε_fd = 1e-3.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def logits_np(self, X):
        """Logit = log(p / (1−p)), score continu utilisé par les attaques."""
        p = np.clip(self.predict_proba(X), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def grad_numerical(self, X, eps_fd=1e-3):
        """Gradient numérique du logit par différences finies centrées. (N, D)."""
        X    = X.astype(np.float64)
        grad = np.zeros_like(X)
        for j in range(X.shape[1]):
            X_plus        = X.copy()
            X_minus       = X.copy()
            X_plus[:, j]  += eps_fd
            X_minus[:, j] -= eps_fd
            grad[:, j] = (self.logits_np(X_plus) - self.logits_np(X_minus)) / (2 * eps_fd)
        return grad

    def grad_bce(self, X, y, eps_fd=1e-3):
        """Approx. du gradient BCE : (p − y) · grad_logit."""
        p        = np.clip(self.predict_proba(X), 1e-7, 1 - 1e-7)
        grad_log = self.grad_numerical(X, eps_fd)
        return (p - y)[:, None] * grad_log

    def grad_logit(self, X, eps_fd=1e-3):
        """Alias vers grad_numerical (cohérence avec LogRegWrapper.grad_logit)."""
        return self.grad_numerical(X, eps_fd)