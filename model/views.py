from django.core.files.storage import default_storage
from django.shortcuts import render
import numpy as np
import pandas as pd
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ARDRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split


def modelDataset(request):
    return render(request, 'webpages/modelDataset.html')
def modelmaking(request):
    if request.method == 'POST':
        global targetvariable
        global file_name
        file = request.FILES['csvfile']
        targetvariable = request.POST['tar']
        file_name = default_storage.save(file.name, file)
    return render(request, 'webpages/model.html')



#All models
def allModelsRegression(request):
    file = default_storage.open(file_name)
    df = pd.read_csv(file)
    col_name = (df.columns.tolist())
    Y = col_name[int(targetvariable)]
    col_name.remove(Y)
    X = col_name
    X = df.drop(Y, axis=1)
    Y = df[Y]
    REGRESSORS = [] 
    REGRESSORS.append(("adaboost", AdaBoostRegressor))
    REGRESSORS.append(("ard_regression", ARDRegression))
    REGRESSORS.append(("decision_tree", DecisionTreeRegressor))
    REGRESSORS.append(("extra_trees", ExtraTreesRegressor ))
    REGRESSORS.append(("gaussian_process", GaussianProcessRegressor))
    REGRESSORS.append(("gradient_boosting", HistGradientBoostingRegressor))
    REGRESSORS.append(("k_nearest_neighbors", KNeighborsRegressor))
    REGRESSORS.append(("libsvm_svr", SVR))
    REGRESSORS.append(("mlp", MLPRegressor))
    REGRESSORS.append(("random_forest", RandomForestRegressor))
    ADJR2 = []
    MAE=[]
    MSE=[]
    RMSE=[]
    names = []
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)  

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)

    for name, model in REGRESSORS:
        try:
            # MEAN ABSOLUTE ERROR
            mae = (-1) * cross_val_score(model(), X_train, y_train,  scoring="neg_mean_absolute_error", cv=5).mean()
            MAE.append(mae)

            #MEAN SQUARE ERROR
            mse = (-1) * cross_val_score(model(), X_train, y_train,  scoring="neg_mean_squared_error", cv=5).mean()
            MSE.append(mse)

            #ROOT MEAN SQUARE ERROR
            rmse=(-1) * cross_val_score(model(), X_train, y_train,  scoring="neg_root_mean_squared_error", cv=5).mean()
            RMSE.append(rmse)

            #R2 Score  
            r_squared = cross_val_score(model(), X, Y,  scoring="r2",cv=3).mean()
            names.append(name)
            n=X_train.shape[0]
            p=X_train.shape[1]
            adj_rsquared=1-(1-r_squared)*((n-1)/(n-p-1))
            ADJR2.append(adj_rsquared)
    
        except Exception as exception:
            print(name + " model failed to execute")
            print(exception)


    data = {}
    for i in range(len(names)):
        temp = {}
        temp2 = []
        temp['adjr'] = ADJR2[i]
        temp['mae'] = MAE[i]
        temp['mse'] = MSE[i]
        temp['rmse'] = RMSE[i]
        temp2.append(temp)
        data[names[i]] = temp2
    
    return render(request, 'webpages/results/resAllModelsRegression.html', data)
def allModelsClassification(request):
    file = default_storage.open(file_name)
    df = pd.read_csv(file)
    col_name = (df.columns.tolist())
    Y = col_name[int(targetvariable)]
    col_name.remove(Y)
    X = col_name
    X = df.drop(Y, axis=1)
    Y = df[Y]
    CLASSIFIERS = []
    CLASSIFIERS.append(("adaboost", AdaBoostClassifier))
    CLASSIFIERS.append(("decision_tree", DecisionTreeClassifier))
    CLASSIFIERS.append(("extra_trees", ExtraTreesClassifier))
    CLASSIFIERS.append(("gaussianNB", GaussianNB))
    CLASSIFIERS.append(("gradient_boosting", HistGradientBoostingClassifier))
    CLASSIFIERS.append(("k_nearest_neighbors", KNeighborsClassifier))
    CLASSIFIERS.append(("libsvm_svc", SVC))
    CLASSIFIERS.append(("mlp", MLPClassifier))
    CLASSIFIERS.append(("random_forest", RandomForestClassifier))
    F1_Score = []
    PRECISION=[]
    RECALL=[]
    ACCURACY=[]
    names = []
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
    for name, model in CLASSIFIERS:
        try:
            precision = cross_val_score(model(), X_train, y_train, cv=5, scoring='precision_weighted',error_score='raise').mean()
            recall = cross_val_score(model(), X_train, y_train, cv=5, scoring='recall_weighted').mean()
            f1 = cross_val_score(model(), X_train, y_train, cv = 5,scoring='f1_weighted').mean()
            accuracy = cross_val_score(model(), X_train, y_train, cv = 5,scoring='accuracy').mean()
            names.append(name)
            F1_Score.append(f1)
            PRECISION.append(precision)
            RECALL.append(recall)
            ACCURACY.append(accuracy)
        except Exception as exception:
            print(name + " model failed to execute")
            print(exception)
    data = {}
    for i in range(len(names)):
        temp = {}
        temp2 = []
        temp['F1_Score'] = F1_Score[i]
        temp['PRECISION'] = PRECISION[i]
        temp['RECALL'] = RECALL[i]
        temp['ACCURACY'] = ACCURACY[i]
        temp2.append(temp)
        data[names[i]] = temp2
    return render(request, 'webpages/results/resAllModelsClassification.html', data)
    


#Regression Models
def Rcommon_imports():
  from sklearn.model_selection import train_test_split
  file = default_storage.open(file_name)
  df = pd.read_csv(file)
  col_name = (df.columns.tolist())
  Y = col_name[int(targetvariable)]
  col_name.remove(Y)
  X = col_name
  X = df.drop(Y, axis=1)
  Y = df[Y]
  X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)  
  return X_train, y_train
def Rresult_metric(model,X,y):
    from sklearn.model_selection import cross_val_score
    r_squared = cross_val_score(model, X, y,  scoring="r2", cv=5).mean() 
    n=X.shape[0]
    p=X.shape[1]
    adj_rsquared=round((1-(1-r_squared)*((n-1)/(n-p-1))),2)
    mae = round(((-1)*cross_val_score(model, X, y,  scoring="neg_mean_absolute_error", cv=5)).mean(),2)
    mse = round(((-1)*cross_val_score(model, X, y,  scoring="neg_mean_squared_error", cv=5)).mean(),2)
    rmse=round(((-1)*cross_val_score(model, X, y,  scoring="neg_root_mean_squared_error", cv=5)).mean(),2)
    return round(adj_rsquared, 3), round(mae, 3), round(mse, 3), round(rmse, 3)

def linear(request):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    X_train, y_train = Rcommon_imports()
    model = LinearRegression()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def ridge(request):
    from sklearn.linear_model import Ridge
    X_train, y_train = common_imports()
    model = Ridge()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def lasso(request):
    from sklearn.linear_model import Lasso
    model = Lasso()
    X_train, y_train = Rcommon_imports()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def enr(request):
    from sklearn.linear_model import ElasticNet
    X_train, y_train = Rcommon_imports()
    model = ElasticNet()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def ard(request):
    from sklearn.linear_model import ARDRegression
    X_train, y_train = Rcommon_imports()
    model = ARDRegression()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def sgd(request):
    from sklearn.linear_model import SGDRegressor
    X_train, y_train = Rcommon_imports()
    model = SGDRegressor()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def svr(request):
    from sklearn.svm import SVR
    X_train, y_train = Rcommon_imports()
    model = SVR()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def dtr(request):
    from sklearn.tree import DecisionTreeRegressor
    X_train, y_train = Rcommon_imports()
    model = DecisionTreeRegressor()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def rfr(request):
    from sklearn.ensemble import RandomForestRegressor
    X_train, y_train = Rcommon_imports()
    model = RandomForestRegressor()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def gbr(request):
    from sklearn.ensemble import GradientBoostingRegressor
    X_train, y_train = Rcommon_imports()
    model = GradientBoostingRegressor()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def lgbm(request):
    from lightgbm import LGBMRegressor
    X_train, y_train = Rcommon_imports()
    model = LGBMRegressor()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def xgbr(request):
    from xgboost.sklearn import XGBRegressor
    X_train, y_train = Rcommon_imports()
    model = XGBRegressor(verbosity=0)
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    print(model)
    return render(request, 'webpages/results/resListRegModels.html', data)
def guassian(request):
    from sklearn.gaussian_process import GaussianProcessRegressor
    X_train, y_train = Rcommon_imports()
    model = GaussianProcessRegressor()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def knr(request):
    from sklearn.neighbors import KNeighborsRegressor
    X_train, y_train = Rcommon_imports()
    model = KNeighborsRegressor()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)
def mlp(request):
    from sklearn.neural_network import MLPRegressor
    X_train, y_train = Rcommon_imports()
    model = MLPRegressor()
    adj_rsquared, mae, mse, rmse = Rresult_metric(model, X_train, y_train)    
    data = {
        'adj_rsquared': round(adj_rsquared, 2),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': (str(model))[:-2]
    }
    return render(request, 'webpages/results/resListRegModels.html', data)



# Classification Models
def common_imports():
    file = default_storage.open(file_name)
    df = pd.read_csv(file)
    col_name = (df.columns.tolist())
    Y = col_name[int(targetvariable)]
    col_name.remove(Y)
    X = col_name
    X = df.drop(Y, axis=1)
    Y = df[Y]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X_train, y_train
def result_metric(model, X, y):
  from sklearn.model_selection import cross_val_score
  precision = cross_val_score(model, X, y, cv=5, scoring='precision_weighted', error_score='raise').mean()
  recall = cross_val_score(model, X, y, cv=5, scoring='recall_weighted').mean()
  f1 = cross_val_score(model, X, y, cv=5, scoring='f1_weighted').mean()
  accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
  return round(precision, 3), round(recall, 3), round(f1, 3), round(accuracy, 3)

def logistic(request):
    from sklearn.linear_model import LogisticRegression
    X_train, y_train = common_imports()
    model = LogisticRegression()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model': "Logistic Classification"
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def svc(request):
    from sklearn.svm import SVC
    X_train, y_train = common_imports()
    model = SVC()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec' : recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def dtc(request):
    from sklearn.tree import DecisionTreeClassifier
    X_train, y_train = common_imports()
    model = DecisionTreeClassifier()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def gaussianNB(request):
    from sklearn.naive_bayes import GaussianNB
    X_train, y_train = common_imports()
    model = GaussianNB()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def multinomialNB(request):
    from sklearn.naive_bayes import MultinomialNB
    X_train, y_train = common_imports()
    model = MultinomialNB()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def sgdc(request):
    from sklearn.linear_model import SGDClassifier
    X_train, y_train = common_imports()
    model = SGDClassifier()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def knnc(request):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    X_train, y_train = common_imports()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def rfc(request):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    X_train, y_train = common_imports()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def gbc(request):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    X_train, y_train = common_imports()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def lgbmc(request):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier()
    X_train, y_train = common_imports()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : str(model)[:-2]
    }
    return render(request, 'webpages/results/resListClassModels.html', data)
def xgbc(request):
    from xgboost.sklearn import XGBClassifier
    model = XGBClassifier()
    X_train, y_train = common_imports()
    precision, recall, f1, accuracy = result_metric(model, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre' : precision,
        'rec': recall,
        'acc': accuracy,
        'model' : "XG Boost Classifier"
    }
    return render(request, 'webpages/results/resListClassModels.html', data)




