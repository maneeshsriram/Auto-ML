from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    
    path('overview', views.overview, name='overview'),
    path('resOverview', views.resOverview, name='resOverview'),

    path('preprocessing', views.preprocessing, name='preprocessing'),
    path('resPreprocessing', views.resPreprocessing, name='resPreprocessing'),

    path('visualization', views.visualization, name='visualization'),
    path('resVisualization', views.resVisualization, name='resVisualization'),
    
    path('parameter', views.parameter, name='parameter'),
    path('deployment', views.deployment, name='deployment'),

    path('predictionDataset', views.predictionDataset, name='predictionDataset'),
    path('prediction', views.prediction, name='prediction'),
]
