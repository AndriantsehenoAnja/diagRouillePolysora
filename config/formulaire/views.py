from django.shortcuts import render,redirect

def index(request):
    if request.method == 'POST':
        nom_saisi = request.POST.get('image', '')
        return redirect(f'/merci/?nom={nom_saisi}')
        
    return render(request, 'formulaire/index.html')

def merci(request):
    nom = request.GET.get('nom', 'Inconnu')
    return render(request, 'formulaire/merci.html', {'nom': nom})