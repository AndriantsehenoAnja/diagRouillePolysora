import cv2
import numpy as np
# import 
def extraire_toutes_caracteristiques(chemin_image):
    # L'image est chargée en BGR par OpenCV
    img = cv2.imread(chemin_image)
    
    # 1. Extraction de couleur (Espace HSV)
    # Étape 1 : Conversion en HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Le reste du code va suivre...
    # Bornes inférieures et supérieures pour la couleur rouille
    lower_rust = np.array([10, 50, 50])   # On baisse S et V pour attraper les marrons sombres
    upper_rust = np.array([30, 255, 255]) # On s'arrête à H=30 pour éviter le vert
    
    # blanc rouille(255) et noir non rouille(0) 
    mask = cv2.inRange(img_hsv, lower_rust, upper_rust)
    nbr_pixel_total = mask.size  # Nombre total de pixels dans l'image
    pct_rouille = (np.count_nonzero(mask)/nbr_pixel_total)*100
    
    # 2. Extraction de texture et rugosité (Filtre de Sobel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # . Calcul des gradients X et Y (en utilisant un format de données 64-bit pour éviter les pertes)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # . Combinaison des deux pour avoir la magnitude du gradient
    sobel_combined = cv2.magnitude(sobelx, sobely)
    #variance (np.var()) est souvent très efficace pour la rugosité 
    # car elle mesure à quel point les valeurs s'écartent de
    # la moyenne (ce qui traduit bien le côté "irrégulier" des pustules).
    rugosite = np.var(sobel_combined)
    
    # le variable de mon choix(circulation moyenne)
    # comme les rouilles sont de la forme circulaire ovale donc on peut utiliser la circularité pour détecter les rouilles
    # 
    # 1. Trouver tous les contours des taches dans le masque de rouille
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Compter le nombre de taches détectées (X4)
    # nombre_taches = len(contours)

    # 3. Analyser la circularité moyenne des taches (X5)
    circularites = []
    for c in contours:
        aire = cv2.contourArea(c)
        perimetre = cv2.arcLength(c, True)

        # Éviter la division par zéro pour les micro-pixels
        if perimetre > 0:
            # Formule mathématique de la circularité : 4 * pi * Aire / (Périmètre^2)
            # Un cercle parfait a une circularité de 1.0
            circ = (4 * np.pi * aire) / (perimetre ** 2)
            circularites.append(circ)

    # Calculer la circularité moyenne de toutes les taches de l'image
    circularite_moyenne = np.mean(circularites) if circularites else 0.0
    
    return {
        'pct_rouille': pct_rouille,
        'rugosite':rugosite,
        'mon_variable': circularite_moyenne
    }



    
    