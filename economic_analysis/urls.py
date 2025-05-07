from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('projects/', views.projects_page, name='projects_page'),  # Новый путь
    path('calculate/', views.calculate, name='calculate'),  # Путь для расчета
]
