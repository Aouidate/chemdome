from django.shortcuts import render

# Create your views here.


def about_us(request):
    return render(request, 'about_us.html')

def contact_us(request):
    return render(request, 'contact_us.html')

def custom_404_view(request, exception):
    return render(request, '404.html', {}, status=404)

def domain_view(request):
    return render(request, 'pca_domain/templates/pca_domain.html')

def home(request):
    return render(request, 'homepage.html')

def knn_view(request):
    return render(request, 'knn_domain/templates/knn_domain.html')

def pca_view(request):
    return render(request, 'leverage_domain/templates/leverage_domain.html')

def sali(request):
    return render(request, 'sali/templates/sali.html')