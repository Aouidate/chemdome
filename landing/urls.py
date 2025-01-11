# landing/urls.py
from django.urls import path
from django.conf.urls import handler404 #This is to handl errors
from . import views
from leverage_domain import views as domain_views
from pca_domain import views as pca_views
from landing.views import custom_404_view

urlpatterns = [
    path('', views.home, name='home'),
    path('about-us/', views.about_us, name='about_us'),  # About Us page
    path('contact-us/', views.contact_us, name='contact_us'),  # Contact Us page
    path('domain/', domain_views.calculate_domain, name='domain_calculate'),
    path('pca/', pca_views.calculate_domain, name='pca_calculate'),
]
handler404 = custom_404_view