from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('preprocessing', views.preprocessing, name='preprocessing'),
    path('resPreprocessing', views.resPreprocessing, name='resPreprocessing'),

    path('visualization', views.visualization, name='visualization'),
    path('resVisualization', views.resVisualization, name='resVisualization'),
    
    path('model', views.model, name='model'),
    path('parameter', views.parameter, name='parameter'),
    path('prediction', views.prediction, name='prediction'),
    path('deployment', views.deployment, name='deployment'),
]
