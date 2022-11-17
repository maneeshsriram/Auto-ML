from django.urls import path
from . import views

urlpatterns = [
    path('tuningDataset', views.tuningDataset, name='tuningDataset'),
    path('tuningPkl', views.tuningPkl, name='tuningPkl'),
    path('tuningMethod', views.tuningMethod, name='tuningMethod'),
    path('tuningRegression', views.tuningRegression, name='tuningRegression'),
    path('tuningClassification', views.tuningClassification, name='tuningClassification'),

    path('parameterGrid', views.parameterGrid, name='parameterGrid'),
    path('paramDownloadModel', views.paramDownloadModel, name='paramDownloadModel'),
    
    path('parameterRandom', views.parameterRandom, name='parameterRandom'),
    path('paramDownloadModelRandom', views.paramDownloadModelRandom, name='paramDownloadModelRandom'),
]
