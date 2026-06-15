from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('merci/', views.merci, name='merci'),
]