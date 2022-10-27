from django.urls import path
from . import views

urlpatterns = [
    path('modelmaking', views.modelmaking, name='modelmaking'),
    path('allModels', views.allModels, name='allModels'),
    path('chooseModel', views.chooseModel, name='chooseModel'),
]
