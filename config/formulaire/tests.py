import numpy as np
import pickle
# Import de ta fonction OpenCV
from config.formulaire.traitement import extraire_toutes_caracteristiques 

# 1. Charger la forêt que tu as entraînée (si elle a été sauvegardée en .pkl)
# (Si tu l'as déjà en mémoire dans ton script 'a.py', tu peux ignorer ces 2 lignes)
with open('config/formulaire/modele_rouille.pkl', 'rb') as fichier:
    foret = pickle.load(fichier)

# 2. Définir le chemin de ton image téléchargée
chemin_image = "/home/nekena/Images/diagRouillePolysora/Image collée.png" 

print(f" Analyse de l'image : {chemin_image}...")

# 3. Extraction des caractéristiques via OpenCV
features = extraire_toutes_caracteristiques(chemin_image) 

# 4. Formatage pour la forêt : tableau 2D (1 ligne, 3 colonnes)
nouvelle_observation = np.array([[
    features['pct_rouille'], 
    features['rugosite'], 
    features['mon_variable']
]])

print(f"Features extraites : {nouvelle_observation}")

# 5. Prédiction
prediction = foret.predict(nouvelle_observation)

# 6. Interprétation du résultat (on extrait le premier élément [0] de la liste renvoyée)
print("\n--- RÉSULTAT DU DIAGNOSTIC ---")
if prediction[0] == 1:
    print(" Résultat : La feuille est MALADE (Rouille Polysora détectée).")
else:
    print(" Résultat : La feuille est SAINE.")
print("------------------------------")