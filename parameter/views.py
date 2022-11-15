from sklearn.model_selection import train_test_split
from django.shortcuts import render
from django.core.files.storage import default_storage

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
  return precision, recall, f1, accuracy

def Regresult_metric(model,X,y):
  from sklearn.model_selection import cross_val_score
  r_squared = cross_val_score(model, X, y,  scoring="r2", cv=2).mean()
  n=X.shape[0]
  p=X.shape[1]
  adj_rsquared=1-(1-r_squared)*((n-1)/(n-p-1))
  mae = (-1) *cross_val_score(model, X, y,  scoring="neg_mean_absolute_error", cv=2).mean()
  mse = (-1) *cross_val_score(model, X, y,  scoring="neg_mean_squared_error", cv=2).mean()
  rmse= (-1) *cross_val_score(model, X, y,  scoring="neg_root_mean_squared_error", cv=2).mean()
  return adj_rsquared, mae, mse, rmse


# Classification Grid
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
    gridcv = GridSearchCV(estimator=model, param_grid=grid, cv=3, n_jobs=-1, scoring=choice(metric))
    gridcv.fit(X_train, y_train)
    precision, recall, f1, accuracy = result_metric(gridcv, X_train, y_train)
    # return gridcv
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


# Regression Grid
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
    n_estimators = [x for x in range(5, 1000,100)]
    learning_rate=[10**x for x in range(-5,5)]
    loss=['linear','square','exponential']
    grid={
        'n_estimators':n_estimators,
        'learning_rate':learning_rate,
        'loss':loss
          }
    model=AdaBoostRegressor()
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
        return render(request, 'webpages/results/resListRegModels.html', data)
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
    return render(request, 'webpages/results/resListClassModels.html', data)








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
  return precision, recall, f1

# Classification Random
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
  randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                               random_state=100,n_jobs=-1,scoring=choice(metric))

  randomcv.fit(X_train,y_train)
  precision, recall, f1 = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
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
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1 = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
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
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1 = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
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
    randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,cv=3,
                               random_state=100,n_jobs=-1,scoring=choice(metric))

    randomcv.fit(X_train,y_train)
    precision, recall, f1 = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
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
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1 = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
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
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1 = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
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
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                  random_state=100, n_jobs=-1, scoring=choice(metric))

    randomcv.fit(X_train, y_train)
    precision, recall, f1 = result_metric(randomcv, X_train, y_train)
    data = {
        'f1_score': f1,
        'pre': precision,
        'rec': recall,
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
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1 = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
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
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1 = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
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
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1 = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
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
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1 = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
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
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1 = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
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
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))

  randomcv.fit(X_train, y_train)
  precision, recall, f1 = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
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
  randomcv = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
                                random_state=100, n_jobs=-1, scoring=choice(metric))
  randomcv.fit(X_train, y_train)
  precision, recall, f1 = result_metric(randomcv, X_train, y_train)
  data = {
      'f1_score': f1,
      'pre': precision,
      'rec': recall,
      'model': str(model)[:-2]
  }
  return data



#Regression Random
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
    n_estimators = [x for x in range(5, 1000,100)]
    learning_rate=[10**x for x in range(-5,5)]
    loss=['linear','square','exponential']
    random_grid={
        'n_estimators':n_estimators,
        'learning_rate':learning_rate,
        'loss':loss
          }
    model=AdaBoostRegressor()
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
        return render(request, 'webpages/results/resListRegModels.html', data)
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
    return render(request, 'webpages/results/resListClassModels.html', data)


