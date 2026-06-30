import pandas as pd
import numpy as np
import pickle

from config.formulaire.scripts_data import lancer_parcours_dataset
from config.formulaire.DecisionTreeMaxMinority import DecisionTreeMaxMinority
from config.formulaire.RandomForestMaxMinority import RandomForestMaxMinority

# 1. Chargement des données d'imagerie
dataset_final = lancer_parcours_dataset()
df = pd.DataFrame(dataset_final)

X = df[['pct_rouille', 'rugosite', 'mon_variable']].values
y = df['label'].values


indices = np.arange(len(X))
np.random.seed(42) # Permet de garder le même mélange à chaque exécution
np.random.shuffle(indices)

split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


print("\nEntraînement de la forêt aléatoire sur 80% des données...")
foret = RandomForestMaxMinority(n_trees=15, max_depth=4)
foret.fit(X_train, y_train)

# Évaluation sur les données d'entraînement (vu)
predictions_train = foret.predict(X_train)
precision_train = np.mean(predictions_train == y_train) * 100
print(f" Précision Train (Données vues) : {precision_train:.2f}%")

# Évaluation sur les données de test (jamais vues)
predictions_test = foret.predict(X_test)
precision_test = np.mean(predictions_test == y_test) * 100
print(f" Précision Test (Données INCONNUES) : {precision_test:.2f}%")


vrais_positifs = np.sum((predictions_test == 1) & (y_test == 1))
vrais_negatifs = np.sum((predictions_test == 0) & (y_test == 0))
faux_positifs  = np.sum((predictions_test == 1) & (y_test == 0))
faux_negatifs  = np.sum((predictions_test == 0) & (y_test == 1))

print("\n Analyse des erreurs sur le Test (Matrice de Confusion) :")
print(f"   ---- Vrais Malades bien détectés : {vrais_positifs}")
print(f"   ---- Vrais Saines bien détectées : {vrais_negatifs}")
print(f"   ---- Faux Positifs (Saines dites Malades) : {faux_positifs}")
print(f"   ---- Faux Négatifs (Malades ratées !)     : {faux_negatifs} ")

with open('config/formulaire/modele_rouille.pkl', 'wb') as fichier:
    pickle.dump(foret, fichier)
print(" Modèle sauvegardé avec succès dans 'modele_rouille.pkl' !")

