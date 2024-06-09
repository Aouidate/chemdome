from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

from django.shortcuts import render

def home(request):
    return render(request, 'homepage.html')

def pca_view(request):
    return render(request, 'domain/templates/upload.html')

def domain_view(request):
    return render(request, 'pca_app/templates/upload.html')