"""
URL configuration for chemdome project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from leverage_domain import views as domain_views
from pca_domain import views as pca_views
from landing import views as landing_views
from sali import views as sali_views
from knn_domain import views as knn_views
#from domain.views import calculate_domain
#from pca_app.views import calculate_domain


urlpatterns = [
    #path('', landing_views.home, name='home'),
    path('', include('landing.urls')),  # Includes landing page URLs
    path('domain/', domain_views.calculate_domain, name='domain_calculate'),
    path('knn/', knn_views.knn_domain, name='knn_calculate'),
    path('pca/', pca_views.calculate_domain, name='pca_calculate'),
    path('sali/', sali_views.calculate_sali, name='calculate_sali'),


#    path('pca_calculate/', include('pca_domain.urls')),  # PCA app URL
#    path('leverage/', include('leverage_domain.urls')),  # Applicability domain URL
]

#/ urls.py project folder