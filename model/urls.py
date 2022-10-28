from django.urls import path
from . import views

urlpatterns = [
    path('modelmaking', views.modelmaking, name='modelmaking'),
    path('allModels', views.allModels, name='allModels'),

    path('modelList', views.modelList, name='modelList'),
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
]
