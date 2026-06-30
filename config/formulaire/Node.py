class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        # Pour les nœuds de décision (internes)
        self.feature_idx = feature_idx  # Indice de la variable utilisée pour couper
        self.threshold = threshold      # Valeur du seuil de coupure
        self.left = left                # Sous-arbre gauche (X <= seuil)
        self.right = right              # Sous-arbre droite (X > seuil)
        
        # Pour les feuilles (nœuds terminaux)
        self.value = value              # Classe majoritaire prédite (0 ou 1)
        
    def is_leaf(self):
        return self.value is not None