import numpy as np
from config.formulaire.meilleursp import trouver_meilleur_split
from config.formulaire.Node import Node
class DecisionTreeMaxMinority:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        # Convertir en tableaux numpy au cas où
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # --- Critères d'arrêt ---
        # 1. Nœud 100% pur
        if n_labels == 1:
            return Node(value=y[0])
        # 2. Profondeur maximale atteinte ou plus assez d'échantillons
        if depth >= self.max_depth or n_samples <= 1:
            valeur_majoritaire = int(np.round(np.mean(y)))
            return Node(value=valeur_majoritaire)

        # --- Recherche du meilleur split ---
        meilleur_gain = -1.0
        meilleure_feature = None
        meilleur_seuil = None

        # On teste toutes les colonnes (variables)
        for idx in range(n_features):
            X_column = X[:, idx]
            seuil, gain = trouver_meilleur_split(X_column, y)
            
            if seuil is not None and gain > meilleur_gain:
                meilleur_gain = gain
                meilleure_feature = idx
                meilleur_seuil = seuil

        # Si aucun split n'améliore la pureté, on fait une feuille
        if meilleur_gain <= 0:
            valeur_majoritaire = int(np.round(np.mean(y)))
            return Node(value=valeur_majoritaire)

        # --- Séparation des données et récursion ---
        indices_gauche = X[:, meilleure_feature] <= meilleur_seuil
        indices_droite = X[:, meilleure_feature] > meilleur_seuil

        gauche = self._build_tree(X[indices_gauche], y[indices_gauche], depth + 1)
        droite = self._build_tree(X[indices_droite], y[indices_droite], depth + 1)

        return Node(feature_idx=meilleure_feature, threshold=meilleur_seuil, left=gauche, right=droite)

    def predict_row(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.predict_row(node.left, x)
        return self.predict_row(node.right, x)

    def predict(self, X):
        X = np.array(X)
        return np.array([self.predict_row(self.root, x) for x in X])