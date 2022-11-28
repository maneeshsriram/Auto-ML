from django.urls import path
from . import views

urlpatterns = [
    path('preprocessingForm', views.preprocessingForm, name='preprocessingForm'),
    path('preprocessingCol', views.preprocessingCol, name='preprocessingCol'),
    path('resPreprocessing', views.resPreprocessing, name='resPreprocessing'),
    path('preprocessingDownload', views.preprocessingDownload, name='preprocessingDownload'),

    path('preprocessingOutlier', views.preprocessingOutlier, name='preprocessingOutlier'),
    path('resPreprocessingOutlier', views.resPreprocessingOutlier, name='resPreprocessingOutlier'),

    path('preprocessingFeatureScaling', views.preprocessingFeatureScaling, name='preprocessingFeatureScaling'),
    path('resPreprocessingFeatureScaling', views.resPreprocessingFeatureScaling, name='resPreprocessingFeatureScaling'),

    path('preprocessingFeatureEncoding', views.preprocessingFeatureEncoding, name='preprocessingFeatureEncoding'),
    path('resPreprocessingFeatureEncoding', views.resPreprocessingFeatureEncoding, name='resPreprocessingFeatureEncoding'),

    path('preprocessingFeatureSelection', views.preprocessingFeatureSelection, name='preprocessingFeatureSelection'),
    path('resPreprFV', views.resPreprFV, name='resPreprFV'),
    path('resPreprFC', views.resPreprFC, name='resPreprFC'),
    path('resPreprFMC', views.resPreprFMC, name='resPreprFMC'), 
    path('resPreprFMR', views.resPreprFMR, name='resPreprFMR'),
    path('resPreprFA', views.resPreprFA, name='resPreprFA'),
    path('resPreprER', views.resPreprER, name='resPreprER'),
    path('resPreprEL', views.resPreprEL, name='resPreprEL'),
]
