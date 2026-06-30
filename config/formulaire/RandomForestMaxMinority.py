import numpy as np

from config.formulaire.DecisionTreeMaxMinority import DecisionTreeMaxMinority 
class RandomForestMaxMinority:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_trees):
            # --- Étape du Bagging ---
            # np.random.choice génère des indices aléatoires avec remplacement
            indices_bootstrap = np.random.choice(n_samples, size=n_samples, replace=True)
            X_b = X[indices_bootstrap]
            y_b = y[indices_bootstrap]

            # Entraînement d'un arbre individuel
            arbre = DecisionTreeMaxMinority(max_depth=self.max_depth)
            arbre.fit(X_b, y_b)
            self.trees.append(arbre)

    def predict(self, X):
        # Récupérer les prédictions de chaque arbre : forme (n_arbres, n_echantillons)
        predictions_arbres = np.array([arbre.predict(X) for arbre in self.trees])
        
        # Inversion des axes pour grouper par échantillon : forme (n_echantillons, n_arbres)
        predictions_arbres = np.swapaxes(predictions_arbres, 0, 1)
        
        # --- Vote majoritaire ---
        # On calcule la moyenne des votes pour chaque ligne. Si >= 0.5, la majorité a voté 1.
        predictions_finales = [int(np.round(np.mean(votes))) for votes in predictions_arbres]
        return np.array(predictions_finales)