from django.urls import path
from . import views

urlpatterns = [
    path('preprocessingForm', views.preprocessingForm, name='preprocessingForm'),
    path('preprocessingCol', views.preprocessingCol, name='preprocessingCol'),
    path('resPreprocessing', views.resPreprocessing, name='resPreprocessing'),
    path('preprocessingDownload', views.preprocessingDownload, name='preprocessingDownload'),
    path('preprocessingOutlier', views.preprocessingOutlier, name='preprocessingOutlier'),
    path('resPreprocessingOutlier', views.resPreprocessingOutlier, name='resPreprocessingOutlier'),
]