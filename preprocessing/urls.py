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
    path('preprocFS1', views.preprocFS1, name='preprocFS1'),
    path('preprocFS2', views.preprocFS2, name='preprocFS2'),
    path('preprocFS3', views.preprocFS3, name='preprocFS3'),
    path('resPreprocessingFeatureSelection', views.resPreprocessingFeatureSelection, name='resPreprocessingFeatureSelection'),
]