from django.urls import path
from . import views

urlpatterns = [
    path('formDataset', views.formDataset, name='formDataset'),
    path('PipOverview', views.PipOverview, name='PipOverview'),
    path('PipPreprocessing', views.PipPreprocessing, name='PipPreprocessing'),
]
