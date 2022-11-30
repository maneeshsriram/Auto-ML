from django.urls import path
from . import views

urlpatterns = [
    path('formDataset', views.formDataset, name='formDataset'),
    path('formMetrics', views.formMetrics, name='formMetrics'),

    path('RegMAE', views.RegMAE, name='RegMAE'),
    path('RegMSE', views.RegMSE, name='RegMSE'),
    path('RegRMSE', views.RegRMSE, name='RegRMSE'),
    path('RegAdjR', views.RegAdjR, name='RegAdjR'),
    path('ClassAcc', views.ClassAcc, name='ClassAcc'),
    path('ClassPrec', views.ClassPrec, name='ClassPrec'),
    path('ClassRec', views.ClassRec, name='ClassRec'),
    path('ClassF1', views.ClassF1, name='ClassF1'),

    
]
