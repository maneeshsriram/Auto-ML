from sklearn.model_selection import train_test_split
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import HttpResponse
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")


def tuningDataset(request):
    return render(request, 'webpages/parameterDataset.html')
def tuningPkl(request):
    if request.method == 'POST':
        global targetvariable
        global Dataset_file_name
        targetvariable = request.POST['tar']
        file1 = request.FILES['csvfile']
        Dataset_file_name = default_storage.save(file1.name, file1)
    return render(request, 'webpages/parameterPkl.html')
def tuningRegression(request):
    global regFlag
    regFlag = True
    return render(request, 'webpages/parameterRegression.html')
def tuningClassification(request):
    global regFlag
    regFlag = False
    return render(request, 'webpages/parameterClassification.html')
def tuningMethod(request):
    global model
    global metric 
    model = request.POST['mod']
    metric = request.POST['met']
    return render(request, 'webpages/parameterMethod.html')









def common_imports():
    file = default_storage.open(Dataset_file_name)
    df = pd.read_csv(file)
    col_name = (df.columns.tolist())
    Y = col_name[int(targetvariable)]
    col_name.remove(Y)
    X = col_name
    X = df.drop(Y, axis=1)
    Y = df[Y]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X_train, y_train
def choice(metric):
    choice = {
        'F1': 'f1_weighted',
        'Pre': 'precision_weighted',
        'Rec': 'recall_weighted',
        'Acc': 'accuracy'
    }
    return choice[metric]
def Regchoice(metric):
    choice={
      'AdjR2':'r2',
      'MAE':'neg_mean_absolute_error',
      'MSE':'neg_mean_squared_error',
      'RMSE':'neg_root_mean_squared_error'
    }
    return choice[metric]
def result_metric(model, X, y):
  from sklearn.model_selection import cross_val_score
  precision = cross_val_score(model, X, y, cv=2, scoring='precision_weighted', error_score='raise').mean()
  recall = cross_val_score(model, X, y, cv=2, scoring='recall_weighted').mean()
  f1 = cross_val_score(model, X, y, cv=2, scoring='f1_weighted').mean()
  accuracy = cross_val_score(model, X, y, cv=2, scoring='accuracy').mean()
  return round(precision, 3), round(recall, 3), round(f1, 3), round(accuracy, 3)
def Regresult_metric(model,X,y):
  from sklearn.model_selection import cross_val_score
  r_squared = cross_val_score(model, X, y,  scoring="r2", cv=2).mean()
  n=X.shape[0]
  p=X.shape[1]
  adj_rsquared=1-(1-r_squared)*((n-1)/(n-p-1))
  mae = (-1) *cross_val_score(model, X, y,  scoring="neg_mean_absolute_error", cv=2).mean()
  mse = (-1) *cross_val_score(model, X, y,  scoring="neg_mean_squared_error", cv=2).mean()
  rmse= (-1) *cross_val_score(model, X, y,  scoring="neg_root_mean_squared_error", cv=2).mean()
  return round(adj_rsquared, 3), round(mae, 3), round(mse, 3), round(rmse, 3)





def save_model(model):
    import joblib 
    joblib.dump(model, 'filename.pkl')
    f = open("filename.pkl", "rb")
    content_type = 'application/octet-stream'
    file = f.read()
    response = HttpResponse(file, content_type=content_type)
    response['Content-Disposition'] = 'attachment; filename="model.pkl"'
    return response 
def save_modelGenetic(model):
    import joblib
    joblib.dump({"template": model.fitted_pipeline_,"_imputed": model._imputed}, 'filename.pkl')
    f = open("filename.pkl", "rb")
    content_type = 'application/octet-stream'
    file = f.read()
    response = HttpResponse(file, content_type=content_type)
    response['Content-Disposition'] = 'attachment; filename="model.pkl"'
    return response
def paramDownloadModel(request):
    response = save_model(gridcv)
    return response
def paramDownloadModelRandom(request):
    response = save_model(randomcv)
    return response


#Grid Classification
def logistic_cls_grid(metric):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    X_train, y_train = common_imports()
    penalty = ['l1', 'l2', 'elasticnet', 'none']
    solver = ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']
    max_iter = [10*x for x in range(1, 10)]
    grid = {
        'penalty': penalty,
        'solver': solver,
        'max_iter': max_iter
    }
    model = LogisticRegression()
    global gridcv
    gridcv = GridSearchCV(estimator=model, param_grid=grid, cv=3, n_jobs=-1, scoring=choice(metric))
    gridcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def multinomial_cls_grid(metric):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import GridSearchCV
    X_train, y_train = common_imports()
    alpha = [10**x for x in range(-10, 10)]
    grid = {
        'alpha': alpha
    }
    model = MultinomialNB()
    global gridcv
    gridcv = GridSearchCV(estimator=model, param_grid=grid,
                          cv=3, n_jobs=-1, scoring=choice(metric))
    gridcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def sgd_cls_grid(metric):
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    X_train, y_train = common_imports()
    loss = ["hinge", "log_loss", "log", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber",
            "epsilon_insensitive", "squared_epsilon_insensitive"]
    penalty = ["l2", "l1", "elasticnet"]
    alpha = [10**x for x in range(-10, 10)]
    l1_ratio = [10**x for x in range(-5, 5)]
    max_iter = [500*x for x in range(1, 10)]
    learning_rate = ['constant', 'optimal', 'adaptive', 'invscaling']

    grid = {
        'alpha': alpha,
        'max_iter': max_iter,
        'l1_ratio': l1_ratio,
        'penalty': penalty,
        'loss': loss,
        'learning_rate': learning_rate

    }
    model = SGDClassifier()
    global gridcv
    gridcv = GridSearchCV(estimator=model, param_grid=grid,
                          cv=3, n_jobs=-1, scoring=choice(metric))
    gridcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def lgbm_cls_grid(metric):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV
    X_train, y_train = common_imports()
    num_leaves = [x for x in range(10, 100, 20)]
    learning_rate = [10**x for x in range(-5, 5)]
    n_estimators = [x for x in range(10, 500, 100)]
    min_child_samples = [x for x in range(10, 100, 20)]

    grid = {
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'min_child_samples': min_child_samples
    }
    model = LGBMClassifier()
    global gridcv
    gridcv = GridSearchCV(estimator=model, param_grid=grid,
                          cv=3, n_jobs=-1, scoring=choice(metric))
    gridcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def xgb_cls_grid(metric):
    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    X_train, y_train = common_imports()
    learning_rate = [10**x for x in range(-10, 5)]
    max_depth = [x for x in range(5, 100, 20)]
    min_child_weight = [x for x in range(2, 20, 2)]
    subsample = [0.1*x for x in range(1, 10)]
    colsample_bytree = [0.1*x for x in range(1, 10)]
    n_estimators = [x for x in range(5, 500, 50)]
    grid = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        "min_child_weight": min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    model = XGBClassifier()
    global gridcv
    gridcv = GridSearchCV(estimator=model, param_grid=grid,
                          cv=3, n_jobs=-1, scoring=choice(metric))
    gridcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def adaboost_cls_grid(metric):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    X_train, y_train = common_imports()
    n_estimators = [x for x in range(100, 1000, 100)]
    learning_rate = [10**x for x in range(-5, 5)]

    grid = {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators
    }
    model = AdaBoostClassifier()
    global gridcv
    gridcv = GridSearchCV(estimator=model, param_grid=grid,
                          cv=3, n_jobs=-1, scoring=choice(metric))
    gridcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def decisionTree_cls_grid(metric):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    X_train, y_train = common_imports()
    splitter = ['best', 'random']
    max_depth = [x for x in range(2, 32, 2)]
    min_samples_split = [0.1*x for x in range(1, 10)]
    min_samples_leaf = [0.1*x for x in range(1, 5)]
    max_features = ['auto', 'sqrt', 'log2']
    criterion = ['gini', 'entropy', 'log_loss']

    grid = {'splitter': splitter,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'criterion': criterion
            }
    model = DecisionTreeClassifier()
    global gridcv
    gridcv = GridSearchCV(estimator=model, param_grid=grid,
                          cv=3, n_jobs=-1, scoring=choice(metric))
    gridcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def ExtraTree_cls_grid(metric):
  from sklearn.ensemble import ExtraTreesClassifier
  from sklearn.model_selection import GridSearchCV
  X_train, y_train = common_imports()

  n_estimators = [x for x in range(5, 100, 10)]
  max_depth = [x for x in range(2, 32, 2)]
  min_samples_split = [0.1*x for x in range(1, 10, 2)]
  min_samples_leaf = [0.1*x for x in range(1, 5)]
  max_features = ['auto', 'sqrt', 'log2']
  criterion = ['gini', 'entropy', 'log_loss']

  grid = {'n_estimators': n_estimators,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf,
          'max_features': max_features,
          'criterion': criterion
          }
  model = ExtraTreesClassifier()
  global gridcv
  gridcv = GridSearchCV(estimator=model, param_grid=grid,
                        cv=3, n_jobs=-1, scoring=choice(metric))
  gridcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def gaussian_cls_grid(metric):
  from sklearn.naive_bayes import GaussianNB
  from sklearn.model_selection import GridSearchCV
  X_train, y_train = common_imports()

  var_smoothing = [10**x for x in range(-10, 10)]
  grid = {'var_smoothing': var_smoothing
          }
  model = GaussianNB()
  global gridcv
  gridcv = GridSearchCV(estimator=model, param_grid=grid,
                        cv=3, n_jobs=-1, scoring=choice(metric))
  gridcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def histGradient_cls_grid(metric):
  from sklearn.ensemble import HistGradientBoostingClassifier
  from sklearn.model_selection import GridSearchCV
  X_train, y_train = common_imports()

  learning_rate = [10**x for x in range(-5, 5)]
  max_iter = [x for x in range(5, 100, 20)]
  max_leaf_nodes = [x for x in range(5, 100, 20)]
  max_depth = [x for x in range(1, 5)]
  min_samples_leaf = [x for x in range(1, 10)]
  max_bins = [x for x in range(5, 250, 50)]
  loss = ['log_loss', 'auto', 'binary_crossentropy', 'categorical_crossentropy']

  grid = {'learning_rate': learning_rate,
          'max_iter': max_iter,
          'max_leaf_nodes': max_leaf_nodes,
          'max_depth': max_depth,
          'min_samples_leaf': min_samples_leaf,
          'max_bins': max_bins,
          'loss': loss
          }
  model = HistGradientBoostingClassifier()
  global gridcv
  gridcv = GridSearchCV(estimator=model, param_grid=grid,
                        cv=3, n_jobs=-1, scoring=choice(metric))
  gridcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def KNeighbors_cls_grid(metric):
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import GridSearchCV
  X_train, y_train = common_imports()

  n_neighbors = [x for x in range(1, 100, 10)]
  weights = ['uniform', 'distance']
  algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
  leaf_size = [x for x in range(1, 100, 10)]
  p = [1, 2]

  grid = {'n_neighbors': n_neighbors,
          'weights': weights,
          'algorithm': algorithm,
          'leaf_size': leaf_size,
          'p': p,
          }
  model = KNeighborsClassifier()
  global gridcv
  gridcv = GridSearchCV(estimator=model, param_grid=grid,
                        cv=3, n_jobs=-1, scoring=choice(metric))
  gridcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def svc_reg_grid(metric):
  from sklearn.svm import SVC
  from sklearn.model_selection import GridSearchCV
  X_train, y_train = common_imports()

  kernel = ['linear', 'poly', 'rbf', 'sigmoid']
  gamma = ['scale', 'auto']
  tol = [10**x for x in range(-5, 5)]
  max_iter = [x for x in range(1, 100, 10)]

  grid = {'kernel': kernel,
          'gamma': gamma,
          'tol': tol,
          'max_iter': max_iter
          }
  model = SVC()
  global gridcv
  gridcv = GridSearchCV(estimator=model, param_grid=grid,
                        cv=3, n_jobs=-1, scoring=choice(metric))
  gridcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def mlp_cls_grid(metric):
  from sklearn.neural_network import MLPClassifier
  from sklearn.model_selection import GridSearchCV
  X_train, y_train = common_imports()

  activation = ['identity', 'logistic', 'tanh', 'relu']
  alpha = [10**x for x in range(-5, 5)]
  max_iter = [x for x in range(500, 3000, 200)]

  grid = {
      'activation': activation,
      'alpha': alpha,
      'max_iter': max_iter
  }
  model = MLPClassifier()
  global gridcv
  gridcv = GridSearchCV(estimator=model, param_grid=grid,
                        cv=3, n_jobs=-1, scoring=choice(metric))
  gridcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def randomForest_cls_grid(metric):
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV
  X_train, y_train = common_imports()

  n_estimators = [x for x in range(10, 100, 10)]
  max_features = ['auto', 'sqrt', 'log2']
  max_depth = [x for x in range(10, 200, 20)]
  min_samples_split = [0.1*x for x in range(1, 10)]
  min_samples_leaf = [0.1*x for x in range(1, 5)]
  max_leaf_nodes = [x for x in range(1, 50, 10)]
  criterion = ['gini', 'entropy', 'log_loss']

  grid = {'n_estimators': n_estimators,
          'max_features': max_features,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf,
          'max_leaf_nodes': max_leaf_nodes,
          'criterion': criterion}

  model = RandomForestClassifier()
  global gridcv
  gridcv = GridSearchCV(estimator=model, param_grid=grid,
                        cv=3, n_jobs=-1, scoring=choice(metric))
  gridcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data


#Grid Regression
def linear_reg_grid(metric='Adjusted R2'):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    fit_intercept=[True,False]
    normalize=[True,False]
    grid={'fit_intercept':fit_intercept,
          'normalize':normalize
          }
    model=LinearRegression()
    global gridcv
    gridcv = GridSearchCV(estimator=model, param_grid=grid,
                          cv=3, n_jobs=-1, scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    save_model(gridcv)
    return data
def ridge_reg_grid(metric='Adjusted R2'):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    alpha=[10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,15000,1000)]
    solver=['auto', 'svd', 'cholesky',  'sparse_cg', 'sag', 'saga']
    tol=[10**x for x in range(-5,5)]

    grid={'alpha':alpha,
          'max_iter':max_iter,
          'solver':solver,
          'tol':tol
          }
    model=Ridge()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def lasso_reg_grid(metric='Adjusted R2'):
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    alpha=[10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,10000,2000)]
    grid={'alpha':alpha,
          'max_iter':max_iter
          }
    model=Lasso()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def elastic_net_reg_grid(metric='Adjusted R2'):
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    alpha=[10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,15000,2000)]
    l1_ratio=[0.1*x for x in range(1,10)]
    grid={'alpha':alpha,
          'max_iter':max_iter,
          'l1_ratio':l1_ratio
          }
    model=ElasticNet()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def sgd_reg_grid(metric='Adjusted R2'):
    from sklearn.linear_model import SGDRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    loss=["squared_error","huber", "epsilon_insensitive","squared_epsilon_insensitive"]
    penalty=["l2", "l1", "elasticnet"]
    alpha=[10**x for x in range(-5,5)]
    learning_rate=["invscaling","constant","optimal","adaptive"]
    max_iter=[x for x in range(200,3000,200)]
    l1_ratio=[0.1*x for x in range(1,10)]
    grid={'alpha':alpha,
          'max_iter':max_iter,
          'l1_ratio':l1_ratio,
          'penalty':penalty,
          'loss':loss,
          'learning_rate':learning_rate
          }
    model=SGDRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def lgbm_reg_grid(metric='Adjusted R2'):
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    num_leaves =[x for x in range(10,100,10)]
    n_estimators =[x for x in range(10,500,100)]
    min_child_samples=[x for x in range(10,100,10)]
    learning_rate=[10**x for x in range(-5,5)]
    grid={
        'num_leaves':num_leaves,
        'n_estimators':n_estimators,
        'min_child_samples':min_child_samples,
          'learning_rate':learning_rate
          }
    model=LGBMRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def xgb_reg_grid(metric='Adjusted R2'):
    from xgboost.sklearn import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    colsample_bytree=[0.1*x for x in range(1,10)]
    n_estimators=[x for x in range(5,500,50)]
    subsample=[0.1*x for x in range(1,10)]
    min_child_weight= [x for x in range(1,20,2)]
    max_depth= [x for x in range(5,100,10)]
    learning_rate= [10**x for x in range(-10,1)]
    grid={
        'learning_rate':learning_rate,
        'max_depth':max_depth,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'n_estimators':n_estimators,
        "min_child_weight":min_child_weight,
        'learning_rate':learning_rate
          }
    model=XGBRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def ard_reg_grid(metric='Adjusted R2'):
    from sklearn.linear_model import ARDRegression
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    n_iter = [x for x in range(5, 2000, 200)]
    alpha_1 = [10**x for x in range(-5,5)]
    alpha_2 = [10**x for x in range(-5,5)]
    lambda_1 = [10**x for x in range(-5,5)]
    lambda_2 = [10**x for x in range(-5,5)]
    grid={
        'n_iter':n_iter,
        'alpha_1':alpha_1,
        'alpha_2':alpha_2,
        'lambda_1':lambda_1,
        'lambda_2':lambda_2
          }
    model=ARDRegression()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def adaboost_reg_grid(metric='Adjusted R2'):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    n_estimators = [x for x in range(100, 1000,100)]
    learning_rate=[10**x for x in range(-5,5)]
    loss=['linear','square','exponential']
    grid={
        'n_estimators':n_estimators,
        'learning_rate':learning_rate,
        'loss':loss
          }
    model=AdaBoostRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def decisiontree_reg_grid(metric='Adjusted R2'):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    max_depth = [x for x in range(2,32,2)]
    min_samples_split = [0.1*x for x in range(1,10)]
    min_samples_leaf=  [0.1*x for x in range(1,5)]
    max_features=['auto', 'sqrt', 'log2']
    criterion=['squared_error', 'friedman_mse', 'absolute_error', 'poisson']

    grid={
        'max_depth':max_depth,
        'min_samples_split':min_samples_split,
        'min_samples_leaf':min_samples_leaf,
        'max_features':max_features,
        'criterion':criterion
          }
    model=DecisionTreeRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def extratree_reg_grid(metric='Adjusted R2'):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    n_estimators = [x for x in range(5, 100,10)]
    max_depth = [x for x in range(1,32,2)]
    min_samples_split = [0.1*x for x in range(1,10)]
    min_samples_leaf=  [0.1*x for x in range(1,5)]
    max_features=['auto', 'sqrt', 'log2']
    criterion=['squared_error', 'friedman_mse', 'absolute_error', 'poisson']

    grid={'n_estimators':n_estimators,
        'max_depth':max_depth,
        'min_samples_split':min_samples_split,
        'min_samples_leaf':min_samples_leaf,
        'max_features':max_features,
        'criterion':criterion
          }
    model=ExtraTreesRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def gaussian_reg_grid(metric='Adjusted R2'):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    alpha = [10**x for x in range(-5,5)]
    n_restarts_optimizer=[0,1,2]
    normalize_y=[False,True]

    grid={'alpha':alpha,
        'n_restarts_optimizer':n_restarts_optimizer,
        'normalize_y':normalize_y
          }
    model=GaussianProcessRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def histgradient_reg_grid(metric='Adjusted R2'):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    learning_rate=[10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,100,10)]
    max_leaf_nodes=[x for x in range(5,100,10)]
    max_depth=[x for x in range(5,100,10)]
    min_samples_leaf= [x for x in range(5,100,10)]
    max_bins=[x for x in range(5,250,10)]
    loss=['squared_error', 'least_squares', 'absolute_error', 'least_absolute_deviation', 'poisson']

    grid={'learning_rate':learning_rate,
        'max_iter':max_iter,
        'max_leaf_nodes':max_leaf_nodes,
        'max_depth':max_depth,
        'min_samples_leaf':min_samples_leaf,
        'max_bins':max_bins,
        'loss':loss
          }
    model=HistGradientBoostingRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def kneighbor_reg_grid(metric='Adjusted R2'):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    n_neighbors=[x for x in range(5,100,10)]
    weights=['uniform','distance']
    algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size=[x for x in range(5,100,10)]
    p=[1,2]

    grid={'n_neighbors':n_neighbors,
        'weights':weights,
        'algorithm':algorithm,
        'leaf_size':leaf_size,
        'p':p
          }
    model=KNeighborsRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def svr_reg_grid(metric='Adjusted r2'):
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    kernel=['linear', 'poly', 'rbf', 'sigmoid']
    gamma=['scale','auto']
    tol = [10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,100,10)]  
    grid={'kernel':kernel,
        'gamma':gamma,
        'tol':tol,
        'max_iter':max_iter
          }
    model=SVR()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def mlp_reg_grid(metric='Adjusted R2'):
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GridSearchCV
    X_train,y_train=common_imports()
    activation=['identity', 'logistic', 'tanh', 'relu']
    alpha = [10**x for x in range(-5,5)]
    max_iter=[x for x in range(500,3000,500)]  
    grid={'activation':activation,
        'alpha':alpha,
        'max_iter':max_iter
          }
    model=MLPRegressor()
    global gridcv
    gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
    gridcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def randomForest_reg_grid(metric='Adjusted R2'):
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import GridSearchCV
  X_train,y_train=common_imports()

  n_estimators = [x for x in range(10, 100,10)]
  max_features = ['auto', 'sqrt','log2']
  max_depth=[x for x in range(5,50,5)]
  min_samples_split = [0.1*x for x in range(1,10)]
  min_samples_leaf=  [0.1*x for x in range(1,5)]
  max_leaf_nodes=[x for x in range(10,50,10)]
  criterion=['squared_error', 'absolute_error', 'poisson']

  # Create the random grid
  grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes':max_leaf_nodes,
               'criterion':criterion}

 
  model=RandomForestRegressor()
  global gridcv
  gridcv=GridSearchCV(estimator=model,param_grid=grid,cv=3,n_jobs=-1,scoring=Regchoice(metric))
  gridcv.fit(X_train,y_train)
  adj_rsquared, mae, mse, rmse = Regresult_metric(gridcv, X_train, y_train)
  data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
  return data
def parameterGrid(request):
    if regFlag:
        if model == 'linear':
            data = linear_reg_grid(metric)
        if model == 'ridge':
            data = ridge_reg_grid(metric)
        if model == 'lasso':
            data = lasso_reg_grid(metric)
        if model == 'ElasticNet':
            data = elastic_net_reg_grid(metric)
        if model == 'SGD':
            data = sgd_reg_grid(metric)
        if model == 'LGBM':
            data = lgbm_reg_grid(metric)
        if model == 'XGB':
            data = xgb_reg_grid(metric)
        if model == 'ARDRegression':
            data = ard_reg_grid(metric)
        if model == 'AdaBoost':
            data = adaboost_reg_grid(metric)
        if model == 'DecisionTree':
            data = decisiontree_reg_grid(metric)
        if model == 'ExtraTrees':
            data = extratree_reg_grid(metric)
        if model == 'GaussianProcess':
            data = gaussian_reg_grid(metric)
        if model == 'HistGradientBoosting':
            data = histgradient_reg_grid(metric)
        if model == 'KNeighbors':
            data = kneighbor_reg_grid(metric)
        if model == 'SVR':
            data = svr_reg_grid(metric)
        if model == 'MLP':
            data = mlp_reg_grid(metric)
        if model == 'RandomForest':
            data = randomForest_reg_grid(metric)
        return render(request, 'webpages/results/resParamGridList.html', data)
    else:
        if model == 'Logistic':
            data = logistic_cls_grid(metric) 
        elif model == 'MultinomialNB':
            data = multinomial_cls_grid(metric)
        elif model == 'SGD':
            data = sgd_cls_grid(metric)
        elif model == 'LGBM':
            data = lgbm_cls_grid(metric)
        elif model == 'XGB':
            data = xgb_cls_grid(metric)
        elif model == 'AdaBoost':
            data = adaboost_cls_grid(metric)
        elif model == 'DecisionTree':
            data = decisionTree_cls_grid(metric)
        elif model == 'ExtraTrees':
            data = ExtraTree_cls_grid(metric)
        elif model == 'GaussianNB':
            data = gaussian_cls_grid(metric)
        elif model == 'HistGradientBoosting':
            data = histGradient_cls_grid(metric)
        elif model == 'KNeighbors':
            data = KNeighbors_cls_grid(metric)
        elif model == 'SVC':
            data = svc_reg_grid(metric)
        elif model == 'MLP':
            data = mlp_cls_grid(metric)
        elif model == 'RandomForest':
            data = randomForest_cls_grid(metric)
    return render(request, 'webpages/results/resParamGridClass.html', data)








def Ranchoice(metric):
    choice ={
      'F1': 'f1_weighted',
      'Pre': 'precision_weighted',
      'Rec': 'recall_weighted'
    }
    return choice[metric]
def Ranresult_metric(model,X,y):
  from sklearn.model_selection import cross_val_score
  precision = cross_val_score(model, X, y, cv=2, scoring='precision_weighted',error_score='raise').mean()
  recall = cross_val_score(model, X, y, cv=2, scoring='recall_weighted').mean()
  f1 = cross_val_score(model, X, y, cv = 2,scoring='f1_weighted').mean()
  return round(precision, 3), round(recall, 3), round(f1, 3)

#Random Classification
def logistic_cls_random(metric='F1 Score'):
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import RandomizedSearchCV
  X_train,y_train=common_imports()
  penalty=['l1', 'l2', 'elasticnet', 'none']
  solver=['lbfgs','newton-cg','liblinear','sag','saga']
  max_iter=[500*x for x in range(1,10)]
  random_grid = {
            'penalty': penalty,
            'solver':solver,
            'max_iter':max_iter
          }
  model=LogisticRegression()
  global randomcv
  randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                               random_state=100,n_jobs=-1,scoring=choice(metric))

  randomcv.fit(X_train,y_train)
  precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def multinomial_cls_random(metric='F1 Score'):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import RandomizedSearchCV
    X_train, y_train = common_imports()
    alpha = [10**x for x in range(-10, 10)]

    random_grid = {
        'alpha': alpha
    }
    model = MultinomialNB()
    global randomcv
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def sgdclassifier_cls_random(metric='F1 Score'):
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import RandomizedSearchCV
    X_train, y_train = common_imports()
    loss = ["hinge", "log_loss", "log", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber",
            "epsilon_insensitive", "squared_epsilon_insensitive"]
    penalty = ["l2", "l1", "elasticnet"]
    alpha = [10**x for x in range(-5, 5)]
    l1_ratio = [0.1*x for x in range(1, 10)]
    max_iter = [x for x in range(200, 3000, 200)]
    learning_rate = ['constant', 'optimal', 'adaptive', 'invscaling']

    random_grid = {
        'alpha': alpha,
        'max_iter': max_iter,
        'l1_ratio': l1_ratio,
        'penalty': penalty,
        'loss': loss,
        'learning_rate': learning_rate

    }
    model = SGDClassifier()
    global randomcv
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def lgbmclassifier_cls_random(metric='F1 Score'):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    num_leaves =[x for x in range(10,100,10)]
    learning_rate=[10**x for x in range(-5,5)]
    n_estimators =[x for x in range(10,500,10)]
    min_child_samples=[x for x in range(10,100,10)]

    random_grid={
        'num_leaves':num_leaves,
        'learning_rate':learning_rate,
        'n_estimators':n_estimators,
        'min_child_samples':min_child_samples
          }
    model=LGBMClassifier()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                               random_state=100,n_jobs=-1,scoring=choice(metric))

    randomcv.fit(X_train,y_train)
    precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def xgbclassifier_cls_random(metric='F1 Score'):
    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV
    X_train, y_train = common_imports()
    learning_rate = [10**x for x in range(-10, 5)]
    max_depth = [x for x in range(1, 50, 5)]
    min_child_weight = [x for x in range(1, 20, 2)]
    subsample = [0.1*x for x in range(1, 10)]
    colsample_bytree = [0.1*x for x in range(1, 10)]
    n_estimators = [x for x in range(5, 500, 50)]

    random_grid = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        "min_child_weight": min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    model = XGBClassifier()
    global randomcv
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def adaboostclassifier_cls_random(metric='F1 Score'):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import RandomizedSearchCV
    X_train, y_train = common_imports()
    n_estimators = [x for x in range(100, 1000, 100)]
    learning_rate = [10**x for x in range(-5, 5)]

    random_grid = {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators
    }
    model = AdaBoostClassifier()
    global randomcv
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def decisionclassifier_cls_random(metric='F1 Score'):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import RandomizedSearchCV
    X_train, y_train = common_imports()
    splitter = ['best', 'random']
    max_depth = [x for x in range(2, 32, 2)]
    min_samples_split = [0.1*x for x in range(1, 10)]
    min_samples_leaf = [0.1*x for x in range(1, 5)]
    max_features = ['auto', 'sqrt', 'log2']
    criterion = ['gini', 'entropy', 'log_loss']

    random_grid = {'splitter': splitter,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_features': max_features,
                   'criterion': criterion
                   }
    model = DecisionTreeClassifier()
    global randomcv
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
        'acc': accuracy,
        'model': str(model)[:-2]
    }
    return data
def ExtraTree_cls_random(metric='F1 Score'):
  from sklearn.ensemble import ExtraTreesClassifier
  from sklearn.model_selection import RandomizedSearchCV
  X_train, y_train = common_imports()

  n_estimators = [x for x in range(5, 100, 10)]
  max_depth = [x for x in range(2, 32, 2)]
  min_samples_split = [0.1*x for x in range(1, 10)]
  min_samples_leaf = [0.1*x for x in range(1, 5)]
  max_features = ['auto', 'sqrt', 'log2']
  criterion = ['gini', 'entropy', 'log_loss']

  random_grid = {'n_estimators': n_estimators,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'max_features': max_features,
                 'criterion': criterion
                 }
  model = ExtraTreesClassifier()
  global randomcv
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def gaussian_cls_random(metric='F1 Score'):
  from sklearn.naive_bayes import GaussianNB
  from sklearn.model_selection import RandomizedSearchCV
  X_train, y_train = common_imports()

  var_smoothing = [10**x for x in range(-10, 10)]
  random_grid = {'var_smoothing': var_smoothing
                 }
  model = GaussianNB()
  global randomcv
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def histGradient_cls_random(metric='F1 Score'):
  from sklearn.ensemble import HistGradientBoostingClassifier
  from sklearn.model_selection import RandomizedSearchCV
  X_train, y_train = common_imports()

  learning_rate = [10**x for x in range(-5, 5)]
  max_iter = [x for x in range(5, 100, 10)]
  max_leaf_nodes = [x for x in range(1, 50, 10)]
  max_depth = [x for x in range(5, 100, 19)]
  max_bins = [x for x in range(5, 250, 10)]

  random_grid = {'learning_rate': learning_rate,
                 'max_iter': max_iter,
                 'max_leaf_nodes': max_leaf_nodes,
                 'max_depth': max_depth,
                 'max_bins': max_bins
                 }
  model = HistGradientBoostingClassifier()
  global randomcv
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def KNeighbors_cls_random(metric='F1 Score'):
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import RandomizedSearchCV
  X_train, y_train = common_imports()

  n_neighbors = [x for x in range(5, 80, 10)]
  weights = ['uniform', 'distance']
  algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
  leaf_size = [x for x in range(5, 100, 10)]
  p = [1, 2]

  random_grid = {'n_neighbors': n_neighbors,
                 'weights': weights,
                 'algorithm': algorithm,
                 'leaf_size': leaf_size,
                 'p': p,
                 }
  model = KNeighborsClassifier()
  global randomcv
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def svr_reg_random(metric='F1 Score'):
  from sklearn.svm import SVC
  from sklearn.model_selection import RandomizedSearchCV
  X_train, y_train = common_imports()

  kernel = ['linear', 'poly', 'rbf', 'sigmoid']
  gamma = ['scale', 'auto']
  tol = [10**x for x in range(-5, 5)]
  max_iter = [x for x in range(5, 100, 10)]

  random_grid = {'kernel': kernel,
                 'gamma': gamma,
                 'tol': tol,
                 'max_iter': max_iter
                 }
  model = SVC()
  global randomcv
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def mlp_cls_random(metric='F1 Score'):
  from sklearn.neural_network import MLPClassifier
  from sklearn.model_selection import RandomizedSearchCV
  X_train, y_train = common_imports()

  activation = ['identity', 'logistic', 'tanh', 'relu']
  alpha = [10**x for x in range(-5, 5)]
  max_iter = [x for x in range(500, 3000, 200)]

  random_grid = {
      'activation': activation,
      'alpha': alpha,
      'max_iter': max_iter
  }
  model = MLPClassifier()
  global randomcv
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data
def randomForest_cls_random(metric='F1 Score'):
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import RandomizedSearchCV
  X_train, y_train = common_imports()

  n_estimators = [x for x in range(1, 100, 10)]
  max_features = ['auto', 'sqrt', 'log2']
  max_depth = [x for x in range(10, 200, 20)]
  min_samples_split = [0.1*x for x in range(1, 10)]
  min_samples_leaf = [0.1*x for x in range(1, 5)]
  max_leaf_nodes = [x for x in range(1, 50, 10)]
  criterion = ['gini', 'entropy', 'log_loss']

  random_grid = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'max_leaf_nodes': max_leaf_nodes,
                 'criterion': criterion}

  model = RandomForestClassifier()
  global randomcv
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))
  randomcv.fit(X_train, y_train)
  precision, recall, f1, accuracy = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'acc': accuracy,
      'model': str(model)[:-2]
  }
  return data


#Random Regression
def linear_reg_random(metric='Adjusted R2'):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import RandomizedSearchCV
    X_train, y_train = common_imports()
    fit_intercept = [True, False]
    normalize = [True, False]
    random_grid = {'fit_intercept': fit_intercept,
                   'normalize': normalize
                   }
    model = LinearRegression()
    global randomcv
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=Regchoice(metric))
    randomcv.fit(X_train, y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def ridge_reg_random(metric='Adjusted R2'):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import RandomizedSearchCV
    X_train, y_train = common_imports()
    alpha = [10**x for x in range(-5, 5)]
    max_iter = [x for x in range(1, 15000, 1000)]
    random_grid = {'alpha': alpha,
                   'max_iter': max_iter
                   }
    model = Ridge()
    global randomcv
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=Regchoice(metric))

    randomcv.fit(X_train, y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def lasso_reg_random(metric='Adjusted R2'):
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    alpha=[10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,10000,2000)]
    random_grid={'alpha':alpha,
          'max_iter':max_iter
          }
    model=Lasso()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                      random_state=100,n_jobs=-1,scoring=Regchoice(metric))

    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def elastic_net_reg_random(metric='Adjusted R2'):
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    alpha=[10**x for x in range(-5,5)]
    l1_ratio=[0.1*x for x in range(1,10)]
    max_iter=[x for x in range(1,15000,2000)]
    random_grid={'alpha':alpha,
          'max_iter':max_iter,
          'l1_ratio':l1_ratio
          }
    model=ElasticNet()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                      random_state=100,n_jobs=-1,scoring=Regchoice(metric))

    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def sgd_reg_random(metric='Adjusted R2'):
    from sklearn.linear_model import SGDRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    loss=["squared_error","huber", "epsilon_insensitive","squared_epsilon_insensitive"]
    penalty=["l2", "l1", "elasticnet"]
    alpha=[10**x for x in range(-5,5)]
    learning_rate=["invscaling","constant","optimal","adaptive"]
    max_iter=[x for x in range(200,3000,200)]
    l1_ratio=[0.1*x for x in range(1,10)]
    random_grid={'alpha':alpha,
          'max_iter':max_iter,
          'l1_ratio':l1_ratio,
          'penalty':penalty,
          'loss':loss,
          'learning_rate':learning_rate
          }
    model=SGDRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))

    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def lgbm_reg_random(metric='Adjusted R2'):
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    num_leaves =[x for x in range(10,100,10)]
    n_estimators =[x for x in range(10,500,10)]
    min_child_samples=[x for x in range(10,100,10)]
    learning_rate=[10**x for x in range(-5,5)]
    random_grid={
        'num_leaves':num_leaves,
        'n_estimators':n_estimators,
        'min_child_samples':min_child_samples,
          'learning_rate':learning_rate
          }
    model=LGBMRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))

    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def xgb_reg_random(metric='Adjusted R2'):
    from xgboost.sklearn import XGBRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    colsample_bytree=[0.1*x for x in range(1,10)]
    n_estimators=[x for x in range(5,500,50)]
    subsample=[0.1*x for x in range(1,10)]
    min_child_weight= [x for x in range(1,20,2)]
    max_depth= [x for x in range(5,100,10)]
    learning_rate= [10**x for x in range(-10,1)]
    random_grid={
        'learning_rate':learning_rate,
        'max_depth':max_depth,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'n_estimators':n_estimators,
        "min_child_weight":min_child_weight,
        'learning_rate':learning_rate
          }
    model=XGBRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))

    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def ard_reg_random(metric='Adjusted R2'):
    from sklearn.linear_model import ARDRegression
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    n_iter = [x for x in range(5, 2000, 200)]
    alpha_1 = [10**x for x in range(-5,5)]
    alpha_2 = [10**x for x in range(-5,5)]
    lambda_1 = [10**x for x in range(-5,5)]
    lambda_2 = [10**x for x in range(-5,5)]
    random_grid={
        'n_iter':n_iter,
        'alpha_1':alpha_1,
        'alpha_2':alpha_2,
        'lambda_1':lambda_1,
        'lambda_2':lambda_2
          }
    model=ARDRegression()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))

    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def adaboost_reg_random(metric='Adjusted R2'):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    n_estimators = [x for x in range(100, 1000,100)]
    learning_rate=[10**x for x in range(-5,5)]
    loss=['linear','square','exponential']
    random_grid={
        'n_estimators':n_estimators,
        'learning_rate':learning_rate,
        'loss':loss
          }
    model=AdaBoostRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))
    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def decisiontree_reg_random(metric='Adjusted R2'):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    max_depth = [x for x in range(2,32,2)]
    min_samples_split = [0.1*x for x in range(1,10)]
    min_samples_leaf=  [0.1*x for x in range(1,5)]
    max_features=['auto', 'sqrt', 'log2']
    criterion=['squared_error', 'friedman_mse', 'absolute_error', 'poisson']

    random_grid={
        'max_depth':max_depth,
        'min_samples_split':min_samples_split,
        'min_samples_leaf':min_samples_leaf,
        'max_features':max_features,
        'criterion':criterion
          }
    model=DecisionTreeRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))
    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def extratree_reg_random(metric='Adjusted R2'):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    n_estimators = [x for x in range(5, 100,10)]
    max_depth = [x for x in range(2,32,2)]
    min_samples_split = [0.1*x for x in range(1,10)]
    min_samples_leaf=  [0.1*x for x in range(1,5)]
    max_features=['auto', 'sqrt', 'log2']
    criterion=['squared_error', 'friedman_mse', 'absolute_error', 'poisson']

    random_grid={'n_estimators':n_estimators,
        'max_depth':max_depth,
        'min_samples_split':min_samples_split,
        'min_samples_leaf':min_samples_leaf,
        'max_features':max_features,
        'criterion':criterion
          }
    model=ExtraTreesRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))

    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def gaussian_reg_random(metric='Adjusted R2'):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    alpha = [10**x for x in range(-5,5)]
    n_restarts_optimizer=[0,1,2]
    normalize_y=[False,True]

    random_grid={'alpha':alpha,
        'n_restarts_optimizer':n_restarts_optimizer,
        'normalize_y':normalize_y
          }
    model=GaussianProcessRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))
    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def histgradient_reg_random(metric='Adjusted R2'):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    learning_rate=[10**x for x in range(-5,5)]
    max_iter=[x for x in range(5,100,10)]
    max_leaf_nodes=[x for x in range(5,100,10)]
    max_depth=[x for x in range(5,100,10)]
    min_samples_leaf= [x for x in range(5,100,10)]
    max_bins=[x for x in range(5,250,10)]
    loss=['squared_error', 'least_squares', 'absolute_error', 'least_absolute_deviation', 'poisson']

    random_grid={'learning_rate':learning_rate,
        'max_iter':max_iter,
        'max_leaf_nodes':max_leaf_nodes,
        'max_depth':max_depth,
        'min_samples_leaf':min_samples_leaf,
        'max_bins':max_bins,
        'loss':loss
          }
    model=HistGradientBoostingRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))

    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def kneighbor_reg_random(metric='Adjusted R2'):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    n_neighbors=[x for x in range(5,100,10)]
    weights=['uniform','distance']
    algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size=[x for x in range(5,100,10)]
    p=[1,2]

    random_grid={'n_neighbors':n_neighbors,
        'weights':weights,
        'algorithm':algorithm,
        'leaf_size':leaf_size,
        'p':p
          }
    model=KNeighborsRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))
    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def svr_reg_random(metric='Adjusted R2'):
    from sklearn.svm import SVR
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    kernel=['linear', 'poly', 'rbf', 'sigmoid']
    gamma=['scale','auto']
    tol = [10**x for x in range(-5,5)]
    max_iter=[x for x in range(5,100,10)]  
    random_grid={'kernel':kernel,
        'gamma':gamma,
        'tol':tol,
        'max_iter':max_iter
          }
    model=SVR()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))
    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def mlp_reg_random(metric='Adjusted R2'):
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import RandomizedSearchCV
    X_train,y_train=common_imports()
    activation=['identity', 'logistic', 'tanh', 'relu']
    alpha = [10**x for x in range(-5,5)]
    max_iter=[x for x in range(500,3000,500)]  
    random_grid={'activation':activation,
        'alpha':alpha,
        'max_iter':max_iter
          }
    model=MLPRegressor()
    global randomcv
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                    random_state=100,n_jobs=-1,scoring=Regchoice(metric))
    randomcv.fit(X_train,y_train)
    adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)[:-2]
    }
    return data
def randomForest_reg_random(metric='Adjusted R2'):
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import RandomizedSearchCV
  X_train,y_train=common_imports()

  n_estimators = [x for x in range(10, 100,10)]
  max_features = ['auto', 'sqrt','log2']
  max_depth=[x for x in range(1,5)]
  min_samples_split = [0.1*x for x in range(1,10)]
  min_samples_leaf=  [0.1*x for x in range(1,5)]
  max_leaf_nodes=[x for x in range(10,50,10)]
  criterion=['squared_error', 'absolute_error', 'poisson']

  random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes':max_leaf_nodes,
               'criterion':criterion}

 
  model=RandomForestRegressor()
  global randomcv
  randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                                                                      random_state=100,n_jobs=-1,scoring=Regchoice(metric))

  randomcv.fit(X_train,y_train)
  adj_rsquared, mae, mse, rmse = Regresult_metric(randomcv, X_train, y_train)
  data = {
      'adj_rsquared': adj_rsquared,
      'mae': mae,
      'mse': mse,
      'rmse': rmse,
      'model': str(model)[:-2]
  }
  return data
def parameterRandom(request):
    if regFlag:
        if model == 'linear':
            data = linear_reg_random(metric)
        if model == 'ridge':
            data = ridge_reg_random(metric)
        if model == 'lasso':
            data = lasso_reg_random(metric)
        if model == 'ElasticNet':
            data = elastic_net_reg_random(metric)
        if model == 'SGD':
            data = sgd_reg_random(metric)
        if model == 'LGBM':
            data = lgbm_reg_random(metric)
        if model == 'XGB':
            data = xgb_reg_random(metric)
        if model == 'ARDRegression':
            data = ard_reg_random(metric)
        if model == 'AdaBoost':
            data = adaboost_reg_random(metric)
        if model == 'DecisionTree':
            data = decisiontree_reg_random(metric)
        if model == 'ExtraTrees':
            data = extratree_reg_random(metric)
        if model == 'GaussianProcess':
            data = gaussian_reg_random(metric)
        if model == 'HistGradientBoosting':
            data = histgradient_reg_random(metric)
        if model == 'KNeighbors':
            data = kneighbor_reg_random(metric)
        if model == 'SVR':
            data = svr_reg_random(metric)
        if model == 'MLP':
            data = mlp_reg_random(metric)
        if model == 'RandomForest':
            data = randomForest_reg_random(metric)
        return render(request, 'webpages/results/resParamRandomReg.html', data)
    else:
        if model == 'Logistic':
            data = logistic_cls_random(metric)
        elif model == 'MultinomialNB':
            data = multinomial_cls_random(metric)
        elif model == 'SGD':
            data = sgdclassifier_cls_random(metric)
        elif model == 'LGBM':
            data = lgbmclassifier_cls_random(metric)
        elif model == 'XGB':
            data = xgbclassifier_cls_random(metric)
        elif model == 'AdaBoost':
            data = adaboostclassifier_cls_random(metric)
        elif model == 'DecisionTree':
            data = decisionclassifier_cls_random(metric)
        elif model == 'ExtraTrees':
            data = ExtraTree_cls_random(metric)
        elif model == 'GaussianNB':
            data = gaussian_cls_random(metric)
        elif model == 'HistGradientBoosting':
            data = histGradient_cls_random(metric)
        elif model == 'KNeighbors':
            data = KNeighbors_cls_random(metric)
        elif model == 'SVC':
            data = svr_reg_random(metric)
        elif model == 'MLP':
            data = mlp_cls_random(metric)
        elif model == 'RandomForest':
            data = randomForest_cls_random(metric)
    return render(request, 'webpages/results/resParamRandomClass.html', data)






#Genetic Classification 
def GeneticClasschoice(metric):
    choice={
      'F1':'f1_weighted',
      'Pre':'precision_weighted',
      'Rec':'recall_weighted',
      'Acc':'accuracy'
    }
    return choice[metric]
def GeneticClassresult_metric(model,X,y):
    from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
    y_pred = model.predict(X)
    precision= precision_score(y,y_pred, average='weighted')
    recall= recall_score(y,y_pred, average='weighted')
    f1 = f1_score(y,y_pred, average='weighted')
    accuracy= accuracy_score(y,y_pred)
    return precision, recall, f1, accuracy
def parameterDownloadModelGenetic(request):
    response = save_modelGenetic(tpot_classifier)
    return response
def parameterDownloadModelGeneticReg(request):
    response = save_modelGenetic(tpot_regressor)
    return response
def logistic_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train,y_train=common_imports()
    penalty=['l1', 'l2', 'elasticnet', 'none']
    solver=['lbfgs','newton-cg','liblinear','sag','saga']
    max_iter=[500*x for x in range(1,10)]

    grid={
        'penalty': penalty,
        'solver':solver,
        'max_iter':max_iter
          }

    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,n_jobs=-1,random_state=100,
                                 config_dict={'sklearn.linear_model.LogisticRegression': grid}, 
                                 cv = 5, scoring = GeneticClasschoice(metric))
    tpot_classifier.fit(X_train,y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def multinomial_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train,y_train=common_imports()
    alpha=[10**x for x in range(-10,10)]
    grid={
        'alpha':alpha
          }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,n_jobs=-1,random_state=100,
                                 config_dict={'sklearn.naive_bayes.MultinomialNB': grid}, 
                                 cv = 5, scoring = GeneticClasschoice(metric))
    tpot_classifier.fit(X_train,y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def sgd_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    loss = ["hinge", "log_loss", "log", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber",
            "epsilon_insensitive", "squared_epsilon_insensitive"]
    penalty = ["l2", "l1", "elasticnet"]
    alpha = [10**x for x in range(-10, 10)]
    l1_ratio = [10**x for x in range(-5, 5)]
    max_iter = [500*x for x in range(1, 10)]
    learning_rate = ['constant', 'optimal', 'adaptive', 'invscaling']

    grid = {
        'alpha': alpha,
        'max_iter': max_iter,
        'l1_ratio': l1_ratio,
        'penalty': penalty,
        'loss': loss,
        'learning_rate': learning_rate
    }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.linear_model.SGDClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def lgbm_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    num_leaves = [x for x in range(10, 100, 20)]
    learning_rate = [10**x for x in range(-5, 5)]
    n_estimators = [x for x in range(10, 500, 100)]
    min_child_samples = [x for x in range(10, 100, 20)]

    grid = {
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'min_child_samples': min_child_samples
    }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'lightgbm.LGBMClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def xgb_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    learning_rate = [10**x for x in range(-10, 5)]
    max_depth = [x for x in range(5, 100, 20)]
    min_child_weight = [x for x in range(2, 20, 2)]
    subsample = [0.1*x for x in range(1, 10)]
    colsample_bytree = [0.1*x for x in range(1, 10)]
    n_estimators = [x for x in range(5, 500, 50)]

    grid = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'xgboost.sklearn.XGBClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def adaboost_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    n_estimators = [x for x in range(100, 1000, 100)]
    learning_rate = [10**x for x in range(-5, 5)]

    grid = {'n_estimators': n_estimators,
            'learning_rate': learning_rate
            }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.ensemble.AdaBoostClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def decisionTree_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    splitter = ['best', 'random']
    max_depth = [x for x in range(2, 32, 2)]
    min_samples_split = [0.1*x for x in range(1, 10)]
    min_samples_leaf = [0.1*x for x in range(1, 5)]
    max_features = ['auto', 'sqrt', 'log2']
    criterion = ['gini', 'entropy', 'log_loss']

    grid = {'splitter': splitter,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'criterion': criterion
            }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.tree.DecisionTreeClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def extraTree_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    n_estimators = [x for x in range(5, 100, 10)]
    max_depth = [x for x in range(2, 32, 2)]
    min_samples_split = [0.1*x for x in range(1, 10, 2)]
    min_samples_leaf = [0.1*x for x in range(1, 5)]
    max_features = ['auto', 'sqrt', 'log2']
    criterion = ['gini', 'entropy', 'log_loss']

    grid = {'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'criterion': criterion
            }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.ensemble.ExtraTreesClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def gaussian_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    var_smoothing = [10**x for x in range(-10, 10)]

    grid = {'var_smoothing': var_smoothing
            }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.naive_bayes.GaussianNB': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def histGradient_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    learning_rate = [10**x for x in range(-5, 5)]
    max_iter = [x for x in range(5, 100, 20)]
    max_leaf_nodes = [x for x in range(5, 100, 20)]
    max_depth = [x for x in range(1, 5)]
    min_samples_leaf = [x for x in range(1, 10)]
    max_bins = [x for x in range(5, 250, 50)]
    loss = ['log_loss', 'auto', 'binary_crossentropy',
            'categorical_crossentropy']

    grid = {'learning_rate': learning_rate,
            'max_iter': max_iter,
            'max_leaf_nodes': max_leaf_nodes,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'max_bins': max_bins,
            'loss': loss
            }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.ensemble.HistGradientBoostingClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def KNeighbors_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    n_neighbors = [x for x in range(1, 100, 10)]
    weights = ['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size = [x for x in range(1, 100, 10)]
    p = [1, 2]

    grid = {'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'p': p,
            }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.neighbors.KNeighborsClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def svc_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = ['scale', 'auto']
    tol = [10**x for x in range(-5, 5)]
    max_iter = [x for x in range(1, 100, 10)]

    grid = {'kernel': kernel,
            'gamma': gamma,
            'tol': tol,
            'max_iter': max_iter
            }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={'sklearn.svm.SVC': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def mlp_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train = common_imports()
    activation = ['identity', 'logistic', 'tanh', 'relu']
    alpha = [10**x for x in range(-5, 5)]
    max_iter = [x for x in range(500, 3000, 200)]

    grid = {
        'activation': activation,
        'alpha': alpha,
        'max_iter': max_iter
    }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.neural_network.MLPClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data
def randomForest_cls_genetic(metric='F1 Score'):
    from tpot import TPOTClassifier
    X_train, y_train, X_test, y_test = common_imports()
    n_estimators = [x for x in range(10, 100, 10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [x for x in range(10, 200, 20)]
    min_samples_split = [0.1*x for x in range(1, 10)]
    min_samples_leaf = [0.1*x for x in range(1, 5)]
    max_leaf_nodes = [x for x in range(1, 50, 10)]
    criterion = ['gini', 'entropy', 'log_loss']

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_leaf_nodes': max_leaf_nodes,
            'criterion': criterion
            }
    global tpot_classifier
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                     config_dict={
                                         'sklearn.ensemble.RandomForestClassifier': grid},
                                     cv=5, scoring=GeneticClasschoice(metric))
    tpot_classifier.fit(X_train, y_train)
    precision, recall, f1, accuracy = GeneticClassresult_metric(tpot_classifier,X_train,y_train)
    data = {
      'acc': accuracy,
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)
    }
    return data


#Genetic Regression
def GeneticRegchoice(metric):
    choice={
      'AdjR2':'r2',
      'MAE':'neg_mean_absolute_error',
      'MSE':'neg_mean_squared_error',
      'RMSE':'neg_root_mean_squared_error'
    } 
    return choice[metric]
def GeneticRegresult_metric(model, X, y):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from math import sqrt
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)
    n = X.shape[0]
    p = X.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = sqrt(mse)
    return adj_rsquared, mae, mse, rmse
def linear_reg_genetic(metric):
    from tpot import TPOTRegressor
    X_train,y_train=common_imports()
    fit_intercept=[True,False]
    normalize=[True,False]
    grid={'fit_intercept':fit_intercept,
          'normalize':normalize
          }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,n_jobs=-1,random_state=100,
                                 config_dict={'sklearn.linear_model.LinearRegression': grid}, 
                                 cv = 5, scoring = GeneticRegchoice(metric))
    tpot_regressor.fit(X_train,y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def ridge_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train,y_train=common_imports()
    alpha=[10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,15000,1000)]
    solver=['auto', 'svd', 'cholesky',  'sparse_cg', 'sag', 'saga']
    tol=[10**x for x in range(-5,5)]

    grid={'alpha':alpha,
          'max_iter':max_iter,
          'solver':solver,
          'tol':tol
          }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,n_jobs=-1,random_state=100,
                                 config_dict={'sklearn.linear_model.Ridge': grid}, 
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train,y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def lasso_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train,y_train=common_imports()
    alpha=[10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,10000,2000)]

    grid={'alpha':alpha,
          'max_iter':max_iter
          }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,n_jobs=-1,random_state=100,
                                 config_dict={'sklearn.linear_model.Lasso': grid}, 
                                 cv = 5, scoring = GeneticRegchoice(metric))
    tpot_regressor.fit(X_train,y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def elastic_net_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOT_Regressor
    X_train, y_train = common_imports()
    alpha = [10**x for x in range(-5, 5)]
    max_iter = [x for x in range(1, 15000, 2000)]
    l1_ratio = [0.1*x for x in range(1, 10)]
    grid = {'alpha': alpha,
            'max_iter': max_iter,
            'l1_ratio': l1_ratio
            }
    global tpot_regressor
    tpot_regressor = TPOT_Regressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.linear_model.ElasticNet': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def sgd_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    loss = ["squared_error", "huber", "epsilon_insensitive",
            "squared_epsilon_insensitive"]
    penalty = ["l2", "l1", "elasticnet"]
    alpha = [10**x for x in range(-5, 5)]
    learning_rate = ["invscaling", "constant", "optimal", "adaptive"]
    max_iter = [x for x in range(200, 3000, 200)]
    l1_ratio = [0.1*x for x in range(1, 10)]
    grid = {'alpha': alpha,
            'max_iter': max_iter,
            'l1_ratio': l1_ratio,
            'penalty': penalty,
            'loss': loss,
            'learning_rate': learning_rate
            }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.linear_model.SGDRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def lgbm_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    num_leaves = [x for x in range(10, 100, 10)]
    n_estimators = [x for x in range(10, 500, 100)]
    min_child_samples = [x for x in range(10, 100, 10)]
    learning_rate = [10**x for x in range(-5, 5)]
    grid = {
        'num_leaves': num_leaves,
        'n_estimators': n_estimators,
        'min_child_samples': min_child_samples,
        'learning_rate': learning_rate
    }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'lightgbm.LGBMRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def xgb_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    colsample_bytree = [0.1*x for x in range(1, 10)]
    n_estimators = [x for x in range(5, 500, 50)]
    subsample = [0.1*x for x in range(1, 10)]
    min_child_weight = [x for x in range(1, 20, 2)]
    max_depth = [x for x in range(5, 100, 10)]
    learning_rate = [10**x for x in range(-10, 1)]
    grid = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'n_estimators': n_estimators,
        "min_child_weight": min_child_weight,
        'learning_rate': learning_rate
    }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'xgboost.sklearn.XGBRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def ard_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    n_iter = [x for x in range(5, 2000, 200)]
    alpha_1 = [10**x for x in range(-5, 5)]
    alpha_2 = [10**x for x in range(-5, 5)]
    lambda_1 = [10**x for x in range(-5, 5)]
    lambda_2 = [10**x for x in range(-5, 5)]
    grid = {
        'n_iter': n_iter,
        'alpha_1': alpha_1,
        'alpha_2': alpha_2,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2
    }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.linear_model.ARDRegression': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def adaboost_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    n_estimators = [x for x in range(100, 1000, 100)]
    learning_rate = [10**x for x in range(-5, 5)]
    loss = ['linear', 'square', 'exponential']
    grid = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'loss': loss
    }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.ensemble.AdaBoostRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def decisionTree_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    max_depth = [x for x in range(2, 32, 2)]
    min_samples_split = [0.1*x for x in range(1, 10)]
    min_samples_leaf = [0.1*x for x in range(1, 5)]
    max_features = ['auto', 'sqrt', 'log2']
    criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']

    grid = {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'criterion': criterion
    }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.tree.DecisionTreeRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def extraTree_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    max_depth = [x for x in range(2, 32, 2)]
    min_samples_split = [0.1*x for x in range(1, 10)]
    min_samples_leaf = [0.1*x for x in range(1, 5)]
    max_features = ['auto', 'sqrt', 'log2']
    criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']

    grid = {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'criterion': criterion
    }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.ensemble.ExtraTreesRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def gaussian_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    alpha = [10**x for x in range(-5, 5)]
    n_restarts_optimizer = [0, 1, 2]
    normalize_y = [False, True]

    grid = {'alpha': alpha,
            'n_restarts_optimizer': n_restarts_optimizer,
            'normalize_y': normalize_y
            }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.gaussian_process.GaussianProcessRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def hist_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    learning_rate = [10**x for x in range(-5, 5)]
    max_iter = [x for x in range(1, 100, 10)]
    max_leaf_nodes = [x for x in range(5, 100, 10)]
    max_depth = [x for x in range(5, 100, 10)]
    min_samples_leaf = [x for x in range(5, 100, 10)]
    max_bins = [x for x in range(5, 250, 10)]
    loss = ['squared_error', 'least_squares', 'absolute_error',
            'least_absolute_deviation', 'poisson']

    grid = {'learning_rate': learning_rate,
            'max_iter': max_iter,
            'max_leaf_nodes': max_leaf_nodes,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'max_bins': max_bins,
            'loss': loss
            }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.ensemble.HistGradientBoostingRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def kneighbor_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    n_neighbors = [x for x in range(5, 100, 10)]
    weights = ['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size = [x for x in range(5, 100, 10)]
    p = [1, 2]

    grid = {'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'p': p
            }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.neighbors.KNeighborsRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def svr_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train,y_train=common_imports()
    kernel=['linear', 'poly', 'rbf', 'sigmoid']
    gamma=['scale','auto']
    tol = [10**x for x in range(-5,5)]
    max_iter=[x for x in range(1,100,10)]  

    grid={'kernel':kernel,
        'gamma':gamma,
        'tol':tol,
        'max_iter':max_iter
          }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,n_jobs=-1,random_state=100,
                                 config_dict={'sklearn.svm.SVR': grid}, 
                                 cv = 5, scoring = GeneticRegchoice(metric))
    tpot_regressor.fit(X_train,y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def mlp_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    activation = ['identity', 'logistic', 'tanh', 'relu']
    alpha = [10**x for x in range(-5, 5)]
    max_iter = [x for x in range(500, 3000, 500)]
    grid = {'activation': activation,
            'alpha': alpha,
            'max_iter': max_iter
            }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.neural_network.MLPRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def randomForest_reg_genetic(metric='Adjusted R2'):
    from tpot import TPOTRegressor
    X_train, y_train = common_imports()
    n_estimators = [x for x in range(10, 100, 10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [x for x in range(5, 50, 5)]
    min_samples_split = [0.1*x for x in range(1, 10)]
    min_samples_leaf = [0.1*x for x in range(1, 5)]
    max_leaf_nodes = [x for x in range(10, 50, 10)]
    criterion = ['squared_error', 'absolute_error', 'poisson']
    
    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_leaf_nodes': max_leaf_nodes,
            'criterion': criterion
            }
    global tpot_regressor
    tpot_regressor = TPOTRegressor(generations=5, population_size=24, offspring_size=12,
                                   verbosity=2, early_stop=12, n_jobs=-1, random_state=100,
                                   config_dict={
                                       'sklearn.ensemble.RandomForestRegressor': grid},
                                   cv=5, scoring=GeneticRegchoice(metric))
    tpot_regressor.fit(X_train, y_train)
    save_modelGenetic(tpot_regressor)
    adj_rsquared, mae, mse, rmse = GeneticRegresult_metric(tpot_regressor,X_train,y_train)
    data = {
        'adj_rsquared': adj_rsquared,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'model': str(model)
    }
    return data
def parameterGenetic(request):
    if regFlag:
        if model == 'linear':
            data = linear_reg_genetic(metric)
        if model == 'ridge':
            data = ridge_reg_genetic(metric)
        if model == 'lasso':
            data = lasso_reg_genetic(metric)
        if model == 'ElasticNet':
            data = elastic_net_reg_genetic(metric)
        if model == 'SGD':
            data = sgd_reg_genetic(metric)
        if model == 'LGBM':
            data = lgbm_reg_genetic(metric)
        if model == 'XGB':
            data = xgb_reg_genetic(metric)
        if model == 'ARDRegression':
            data = ard_reg_genetic(metric)
        if model == 'AdaBoost':
            data = adaboost_reg_genetic(metric)
        if model == 'DecisionTree':
            data = decisionTree_reg_genetic(metric)         
        if model == 'ExtraTrees':
            data = extraTree_reg_genetic(metric)
        if model == 'GaussianProcess':
            data = gaussian_reg_genetic(metric)
        if model == 'HistGradientBoosting':
            data = hist_reg_genetic(metric)
        if model == 'KNeighbors':
            data = kneighbor_reg_genetic(metric)
        if model == 'SVR':
            data = svr_reg_genetic(metric)
        if model == 'MLP':
            data = mlp_reg_genetic(metric)
        if model == 'RandomForest':
            data = randomForest_reg_genetic(metric)
        return render(request, 'webpages/results/resParamGeneticReg.html', data)
    else:
        if model == 'Logistic':
            data = logistic_cls_genetic(metric)
        elif model == 'MultinomialNB':
            data = multinomial_cls_genetic(metric)
        elif model == 'SGD':
            data = sgd_cls_genetic(metric)
        elif model == 'LGBM':
            data = lgbm_cls_genetic(metric)
        elif model == 'XGB':
            data = xgb_cls_genetic(metric)
        elif model == 'AdaBoost':
            data = adaboost_cls_genetic(metric)
        elif model == 'DecisionTree':
            data = decisionTree_cls_genetic(metric)
        elif model == 'ExtraTrees':
            data = extraTree_cls_genetic(metric)
        elif model == 'GaussianNB':
            data = gaussian_cls_genetic(metric)
        elif model == 'HistGradientBoosting':
            data = histGradient_cls_genetic(metric)
        elif model == 'KNeighbors':
            data = KNeighbors_cls_genetic(metric)
        elif model == 'SVC':
            data = svc_cls_genetic(metric)
        elif model == 'MLP':
            data = mlp_cls_genetic(metric)
        elif model == 'RandomForest':
            data = randomForest_cls_genetic(metric)
    return render(request, 'webpages/results/resParamGeneticClass.html', data)



