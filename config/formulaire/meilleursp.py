import numpy as np

from config.formulaire.scripts_data import lancer_parcours_dataset

def calculer_gini(y):
    """Calcule l'impureté de Gini d'un vecteur de labels y (0 ou 1)."""
    n = len(y)
    if n == 0:
        return 0
    p1 = np.sum(y) / n
    p0 = 1 - p1
    return 1 - (p0**2 + p1**2)

def trouver_meilleur_split(X_column, y):
    # Convertir en tableaux numpy pour faciliter les manipulations
    X_column = np.array(X_column)
    y = np.array(y)
    
    n_total = len(y)
    if n_total <= 1:
        return None, 0.0
    
    # Étape 1 : Trier les données par X_column
    indices_tries = np.argsort(X_column)
    X_trie = X_column[indices_tries]
    y_trie = y[indices_tries]
    
    # Calculer l'impureté de départ (avant le split)
    gini_depart = calculer_gini(y_trie)
    
    # Étape 2 : Initialiser les variables pour suivre le meilleur seuil et la pureté maximale
    meilleur_seuil = None
    max_gain_purete = -1.0
    
    # Étape 3 : Parcourir les seuils candidats
    for i in range(1, n_total):
        # On ne teste un split que si la valeur change (pour éviter de couper entre deux valeurs identiques)
        if X_trie[i] != X_trie[i-1]:
            # Le seuil candidat est le milieu entre deux valeurs consécutives
            seuil_candidat = (X_trie[i] + X_trie[i-1]) / 2.0
            
            # Séparation des labels selon le seuil
            y_gauche = y_trie[:i]
            y_droite = y_trie[i:]
            
            # Calcul des impuretés des nœuds enfants
            gini_gauche = calculer_gini(y_gauche)
            gini_droite = calculer_gini(y_droite)
            
            # Calcul de l'impureté pondérée après le split
            p_gauche = len(y_gauche) / n_total
            p_droite = len(y_droite) / n_total
            gini_split = p_gauche * gini_gauche + p_droite * gini_droite
            
            # Le gain de pureté est la diminution de l'impureté
            gain_purete = gini_depart - gini_split
            
            # Mise à jour si on trouve un meilleur split
            if gain_purete > max_gain_purete:
                max_gain_purete = gain_purete
                meilleur_seuil = seuil_candidat
                
    # Étape 4 : Retourner le seuil optimal et sa pureté associée
    return meilleur_seuil, max_gain_purete

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Lancer le parcours du dataset pour obtenir les données
    dataset = lancer_parcours_dataset()
    
    # Extraire les caractéristiques et les labels
    X_rugosite = [d['rugosite'] for d in dataset]
    y_labels = [d['label'] for d in dataset]
    
    # Trouver le meilleur split pour la caractéristique "rugosite"
    meilleur_seuil, gain_purete = trouver_meilleur_split(X_rugosite, y_labels)
    
    print(f"Meilleur seuil pour 'rugosite': {meilleur_seuil}, Gain de pureté: {gain_purete}")