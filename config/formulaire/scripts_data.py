import os
# ATTENTION : Vérifiez si votre fichier s'appelle traitement.py (sans S) ou traitements.py
from .traitement import extraire_toutes_caracteristiques 

def lancer_parcours_dataset():
    # Par sécurité, on utilise un chemin absolu par rapport à la racine du projet
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "../dataset") 
    
    classes = {
        "malades": 1,
        "saines": 0
    }
    
    dataset_final = []
    id_image = 1  
    
    print(f"--> Début du scan du dossier : {base_dir}")
    
    for nom_dossier, label in classes.items():
        chemin_dossier = os.path.join(base_dir, nom_dossier)
        
        if not os.path.exists(chemin_dossier):
            print(f"⚠️ Le sous-dossier n'existe pas : {chemin_dossier}")
            continue
            
        fichiers = os.listdir(chemin_dossier)
        print(f"📁 Dossier [{nom_dossier}] trouvé avec {len(fichiers)} éléments.")
            
        for nom_fichier in fichiers:
            if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
                chemin_complet = os.path.join(chemin_dossier, nom_fichier)
                
                try:
                    # Extraction des caractéristiques de base
                    features = extraire_toutes_caracteristiques(chemin_complet)
                    
                    if features is not None:
                        donnees_image = {
                            'id_image': id_image,
                            'pct_rouille': features['pct_rouille'],
                            'rugosite': features['rugosite'],
                            'mon_variable': features['mon_variable'], # Utilise la clé retournée par traitement.py
                            'label': label 
                        }
                        dataset_final.append(donnees_image)
                        id_image += 1  
                except Exception as e:
                    print(f"❌ Erreur lors du traitement de {nom_fichier} : {str(e)}")
                    
    print(f"--> Scan terminé. {len(dataset_final)} images ajoutées au dataset final.")                
    return dataset_final