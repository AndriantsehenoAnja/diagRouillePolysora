from django.shortcuts import render,redirect
from .scripts_data import lancer_parcours_dataset
# def index(request):
#     if request.method == 'POST':
#         nom_saisi = request.POST.get('image', '')
#         listeImage = lancer_parcours_dataset()
#         return redirect(f'/merci/?nom={listeImage}')
        
#     return render(request, 'formulaire/index.html')

def index(request):
    # On exécute le script pour récupérer le tableau de caractéristiques
    donnees_dataset = lancer_parcours_dataset()
    
    # On transmet le tableau au template HTML sous le nom 'images_caracteristiques'
    context = {
        'images_caracteristiques': donnees_dataset
    }
    
    return render(request, 'formulaire/index.html', context)
def merci(request):
    nom = request.GET.get('nom', 'Inconnu')
    return render(request, 'formulaire/merci.html', {'nom': nom})