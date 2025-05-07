from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('projects/', views.projects_page, name='projects_page'),  # Путь для ввода данных в JSON
    path('calculate/', views.calculate_page, name='calculate'),  # Путь для отображения результатов расчёта
    path('parameters/', views.parameters_page, name='parameters_page'),  # Путь для страницы с параметрами
    path('enter_projects/', views.enter_projects_page, name='enter_projects_page'),  # Новый путь для ввода через таблицу
]
