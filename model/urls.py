from django.urls import path
from . import views

urlpatterns = [
    path('modelDataset', views.modelDataset, name='modelDataset'),
    path('modelmaking', views.modelmaking, name='modelmaking'),
    
    path('allModelsRegression', views.allModelsRegression, name='allModelsRegression'),
    path('allModelsClassification', views.allModelsClassification, name='allModelsClassification'),

    path('linear', views.linear, name='linear'),
    path('ridge', views.ridge, name='ridge'),
    path('lasso', views.lasso, name='lasso'),
    path('enr', views.enr, name='enr'),
    path('ard', views.ard, name='ard'),
    path('sgd', views.sgd, name='sgd'),
    path('svr', views.svr, name='svr'),
    path('dtr', views.dtr, name='dtr'),
    path('rfr', views.rfr, name='rfr'),
    path('gbr', views.gbr, name='gbr'),
    path('lgbm', views.lgbm, name='lgbm'),
    path('xgbr', views.xgbr, name='xgbr'),
    path('guassian', views.guassian, name='guassian'),
    path('knr', views.knr, name='knr'),
    path('mlp', views.mlp, name='mlp'),

    path('logistic', views.logistic, name='logistic'),
    path('svc', views.svc, name='svc'),
    path('dtc', views.dtc, name='dtc'),
    path('gaussianNB', views.gaussianNB, name='gaussianNB'),
    path('multinomialNB', views.multinomialNB, name='multinomialNB'),
    path('sgdc', views.sgdc, name='sgdc'),
    path('knnc', views.knnc, name='knnc'),
    path('rfc', views.rfc, name='rfc'),
    path('gbc', views.gbc, name='gbc'),
    path('lgbmc', views.lgbmc, name='lgbmc'),
    path('xgbc', views.xgbc, name='xgbc'),
]
