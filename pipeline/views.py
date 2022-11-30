from django.shortcuts import render
import pandas as pd
import csv
import numpy as np
from django.http import HttpResponse


def formDataset(request):
    return render(request, 'webpages/pipeline/formDataset.html')
def formMetrics(request):
    if request.method == 'POST':
        global df
        global targetVariable
        global col_names
        global col_names_num
        global col_names_cat
        file = request.FILES['csvfile']
        df = pd.read_csv(file)
        col_names = df.columns.tolist()
        col_names_all = df.columns.tolist()
        targetVariable = (request.POST['tar'])
        numericaldata = (request.POST['num']).split()
        col_names_num = []
        for i in numericaldata:
            col_names_num.append(col_names_all[int(i)])
        col_names_cat = col_names_all
        for i in col_names_num:
            try:
                col_names_cat.remove(i)
            except:
                pass
    return render(request, 'webpages/pipeline/formMetrics.html')


def RegMAE(request):
    preProcessing()
    modelReg('Mean Absolute Error')
    return downloadCsv()
def RegMSE(request):
    preProcessing()
    modelReg('Mean Squared Error')
    return downloadCsv()
def RegRMSE(request):
    preProcessing()
    modelReg('Root Mean Squared Error')
    return downloadCsv()
def RegAdjR(request):
    preProcessing()
    modelReg('Adjusted R2')
    return downloadCsv()
def ClassAcc(request):
    preProcessing()
    modelClass('Accuracy')
    return downloadCsv()
def ClassPrec(request):
    preProcessing()
    modelClass('Precision')
    return downloadCsv()
def ClassRec(request):
    preProcessing()
    modelClass('Recall')
    return downloadCsv()
def ClassF1(request):
    preProcessing()
    modelClass('F1 Score')
    return downloadCsv()

def downloadCsv():
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Preprocessed.csv"'
    writer = csv.writer(response)
    col_name_upd = df.columns.tolist()
    writer.writerow(col_name_upd)
    rows = df.to_numpy().tolist()
    for i in range(len(rows)):
        writer.writerow(rows[i])
    return response




#Pre Processing
def preProcessing():
    clean_num_cat()
    miss_handle()
    outlier_handle()
    scale_handle()
    encoding_handle()
    feat_sel_handle()
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
def clean_num_cat():
    for col in col_names_num:
        for j in range(df.shape[0]):
            s = df[col][j]
            s_new = ''
            try:
                for i in s:
                    if(i == '1' or i == '2' or i == '3' or i == '4' or i == '5' or i == '6' or i == '7' or i == '8' or i == '9' or i == '0' or i == '.'):
                        s_new = s_new + i
                s_new = float(s_new)
                df[col][j] = s_new
            except:
                pass

    for i in col_names_cat:
        try:
            df[i] = df[i].astype(str)
        except:
            pass
    for j in col_names_cat:
        for i in range(df.shape[0]):
            if(df[j][i] == 'nan' or df[j][i] == 'NaN'):
                df[j][i] = np.nan
def miss_handle():
    temp = []
    df.dropna(thresh=len(col_names)/2, inplace=True)
    df.reset_index(inplace=True, drop=True)
    for i in range(len(col_names)):  # drop column if more than 40%
        if((df[col_names[i]].isnull().sum()*100/df.shape[0]) >= 40):
            df.drop([col_names[i]], axis=1, inplace=True)
            temp.append(i)
            try:
                col_names_num.remove(col_names[i])
            except:
                pass
            try:
                col_names_cat.remove(col_names[i])
            except:
                pass
    col_temp = col_names
    global targetVariable
    targetCol = col_names[int(targetVariable)]
    for i in temp:
        col_names.remove(col_temp[i])
    for i in range(len(col_names)):
        if col_names[i] == targetCol:
            targetVariable = i
    for i in col_names:
        try:
            df[i].fillna(df[i].median(), inplace=True)
        except:
            df[i].fillna(df[i].mode()[0], inplace=True)
def outlier_handle():  # find ouliers using z score and even isolation forest
    for j in range(len(col_names_num)):
        zscore(j)
def zscore(i):
    thres = 3.5
    mean = np.mean(df[col_names_num[i]])
    std = np.std(df[col_names_num[i]])
    upr_bound = mean+3.5*std
    lwr_bound = mean-3.5*std
    size = df.shape[0]
    for j in range(0, size):
        try:
            if (df[col_names_num[i]][j] < lwr_bound or df[col_names_num[i]][j] > upr_bound):
                df.drop(j, inplace=True)
        except:
            pass
    df.reset_index(inplace=True, drop=True)
def scale_handle():
    col_names_num_sh = col_names_num
    try:
        col_names_num_sh.remove(col_names[int(targetVariable)])
    except:
        pass
    for i in range(len(col_names_num_sh)):
        if((df[col_names_num_sh[i]].max()-df[col_names_num_sh[i]].min())!=0): #if denominator not 0 then normalization or else standardisation
            df[col_names_num_sh[i]]=(df[col_names_num_sh[i]]-df[col_names_num_sh[i]].mean())/(df[col_names_num_sh[i]].max()-df[col_names_num_sh[i]].min())
        else:
            df[col_names_num_sh[i]]=(df[col_names_num_sh[i]]-df[col_names_num_sh[i]].mean())/df[col_names_num_sh[i]].std()
def encoding_handle():
    global targetVariable
    col_names_cat_copy = col_names_cat.copy()
    try:
        col_names_cat_copy.remove(col_names[targetVariable])
    except:
        pass
    temp = []
    for i in col_names_cat_copy:
        if(len(df[i].unique())*100/df.shape[0] >= 50):
            df.drop([i], inplace=True, axis=1)
            temp.append(i)
    targetCol = col_names[int(targetVariable)]
    for i in temp:
        col_names_cat.remove(i)
        col_names_cat_copy.remove(i)
        col_names.remove(i)
    for i in range(len(col_names)):
        if col_names[i] == targetCol:
            targetVariable = i
    for i in range(0,len(col_names_cat_copy)):
        if(len(df[col_names_cat_copy[i]].unique())<=2):
            label_encoding(col_names_cat_copy, i)
        elif(len(df[col_names_cat_copy[i]].unique())<40):
            one_hot_encoding(col_names_cat_copy,i)
        else:
            hash_encoding(col_names_cat_copy,i, 20)
def label_encoding(col_names_cat_copy, i):
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    df[col_names_cat_copy[i]+"_label_encoder"]= label_encoder.fit_transform(df[col_names_cat_copy[i]]) #make new column
    df.drop([col_names_cat_copy[i]], axis=1,inplace=True)   #delete the old column 
    try:
        col_names_cat.remove(col_names_cat_copy[i])
    except:
        pass
    global targetVariable
    targetCol = col_names[int(targetVariable)]
    col_names.remove(col_names_cat_copy[i])
    for i in range(len(col_names)):
        if col_names[i] == targetCol:
            targetVariable = i  
def one_hot_encoding(col_names_cat_copy, i):
    dummies = pd.get_dummies(df[col_names_cat_copy[i]], drop_first=True)
    for j in list(dummies.columns):
        df["ohe_"+j]=dummies[j]
    df.drop([col_names_cat_copy[i]], axis=1,inplace=True)   #delete the old column
    try:
        col_names_cat.remove(col_names_cat_copy[i])
    except:
        pass
    global targetVariable
    targetCol = col_names[int(targetVariable)]
    col_names.remove(col_names_cat_copy[i])
    for i in range(len(col_names)):
        if col_names[i] == targetCol:
            targetVariable = i
def hash_encoding(col_names_cat_copy, i,n):
    import category_encoders as ce
    encoder=ce.HashingEncoder(cols=col_names_cat_copy[i],n_components=n)
    data_encoder=encoder.fit_transform(df[col_names_cat_copy[i]])
    col_name_temp=data_encoder.columns.tolist()
    for j in col_name_temp:
        df["he_"+col_names_cat_copy[i]+j]=data_encoder[j]
    df.drop([col_names_cat_copy[i]], axis=1,inplace=True)   #delete the old column
    try:
        col_names_cat.remove(col_names_cat_copy[i])
    except:
        pass
    global targetVariable
    targetCol = col_names[int(targetVariable)]
    col_names.remove(col_names_cat_copy[i])
    for i in range(len(col_names)):
        if col_names[i] == targetCol:
            targetVariable = i
def feat_sel_handle():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    try:
        temp = []
        new_df = df[col_names_num].copy()
        global targetVariable
        new_df[col_names[int(targetVariable)]] = df[col_names[int(targetVariable)]]
        x = new_df.drop(labels=[col_names[int(targetVariable)]], axis=1)
        y = new_df[col_names[int(targetVariable)]]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
        sel.fit(X_train, y_train)
        selected_feat = X_train.columns[(sel.get_support())]
        removed_feats = X_train.columns[(sel.estimator_.coef_ == 0).ravel().tolist()]
        for i in removed_feats:
            df.drop([i], inplace=True, axis=1)
            temp.append(i)
            try:
                    col_names_num.remove(col_names[i])
            except:
                    pass
        col_temp = col_names
        
        targetCol = col_names[int(targetVariable)]
        for i in temp:
            col_names.remove(i)
        for i in range(len(col_names)):
            if col_names[i] == targetCol:
                targetVariable = i
    except:
        pass





#Model making
def choice(metric):
  choice = {
      'F1 Score': 'f1_weighted',
      'Precision': 'precision_weighted',
      'Recall': 'recall_weighted',
      'Accuracy': 'accuracy'
  }
  return metric
def choiceReg(metric):
    choice = {
        'Adjusted R2': 'r2',
        'Mean Absolute Error': 'neg_mean_absolute_error',
        'Mean Squared Error': 'neg_mean_squared_error',
        'Root Mean Squared Error': 'neg_root_mean_squared_error'
    }
    return choice[metric]

def modelClass(metric):
  pd.set_option("display.precision", 4)
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
  PRECISION = []
  RECALL = []
  ACCURACY = []
  names = []
  ModelObject = {}
  X = df.drop([col_names[targetVariable]], axis=1) 
  Y = df[col_names[targetVariable]]
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
  if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
  for name, model in CLASSIFIERS:
    try:
      precision = cross_val_score(model(), X_train, y_train, cv=5, scoring='precision_weighted', error_score='raise').mean()
      recall = cross_val_score(model(), X_train, y_train, cv=5, scoring='recall_weighted').mean()
      f1 = cross_val_score(model(), X_train, y_train, cv=5, scoring='f1_weighted').mean()
      accuracy = cross_val_score(model(), X_train, y_train, cv=5, scoring='accuracy').mean()
      names.append(name)
      F1_Score.append(f1)
      PRECISION.append(precision)
      RECALL.append(recall)
      ACCURACY.append(accuracy)
      ModelObject[name] = model
    except Exception as exception:
      print(name + " model failed to execute")
      print(exception)
  scores = {
      "Model": names,
      "F1 Score": F1_Score,
      'Precision': PRECISION,
      'Recall': RECALL,
      'Accuracy': ACCURACY
  }
  scores = pd.DataFrame(scores)
  scores = scores.sort_values(by=metric, ascending=False)
  TOP3 = [x for x in scores["Model"].head(3)]
  print(TOP3)
  scores = scores.set_index("Model")
  for i in TOP3:
    if i == 'adaboost':
        adaboost_cls_hyperopt(metric)
    elif i == 'decision_tree':
        decisionTree_cls_hyperopt(metric)
    elif i == 'extra_trees':
        extraTree_cls_hyperopt(metric)
    elif i == 'gaussianNB':
        gaussianb_cls_hyperopt(metric)
    elif i == 'gradient_boosting':
        histGradient_cls_hyperopt(metric)
    elif i == 'k_nearest_neighbors':
        kneighbor_cls_hyperopt(metric)
    elif i == 'libsvm_svc':
        svr_cls_hyperopt(metric)
    elif i == 'mlp':
        mlp_cls_hyperopt(metric)
    elif i == 'random_forest':
        rfc_cls_hyperopt(metric)
def modelReg(metric):
  pd.set_option("display.precision", 4)
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
  from sklearn.model_selection import train_test_split, cross_val_score
  REGRESSORS = []
  REGRESSORS.append(("adaboost", AdaBoostRegressor))
  REGRESSORS.append(("ard_regression", ARDRegression))
  REGRESSORS.append(("decision_tree", DecisionTreeRegressor))
  REGRESSORS.append(("extra_trees", ExtraTreesRegressor))
  REGRESSORS.append(("gaussian_process", GaussianProcessRegressor))
  REGRESSORS.append(("gradient_boosting", HistGradientBoostingRegressor))
  REGRESSORS.append(("k_nearest_neighbors", KNeighborsRegressor))
  REGRESSORS.append(("libsvm_svr", SVR))
  REGRESSORS.append(("mlp", MLPRegressor))
  REGRESSORS.append(("random_forest", RandomForestRegressor))
  ADJR2 = []
  MAE = []
  MSE = []
  RMSE = []
  names = []
  ModelObject = {}
  X = df.drop([col_names[targetVariable]], axis=1)
  Y = df[col_names[targetVariable]]
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
  if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
  for name, model in REGRESSORS:
    try:
      mae = round(-cross_val_score(model(), X_train, y_train, scoring="neg_mean_absolute_error", cv=3).mean(), 2)
      MAE.append(mae)
      mse = round(-cross_val_score(model(), X_train, y_train, scoring="neg_mean_squared_error", cv=3).mean(), 2)
      MSE.append(mse)
      rmse = round(-cross_val_score(model(), X_train, y_train, scoring="neg_root_mean_squared_error", cv=3).mean(), 2)
      RMSE.append(rmse)
      r_squared = round(cross_val_score(model(), X_train, y_train,  scoring="r2", cv=3).mean(), 2)
      names.append(name)
      n = np.array(X_train).shape[0]
      p = np.array(X_train).shape[1]
      adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
      ADJR2.append(adj_rsquared)
      ModelObject[name] = model
    except Exception as exception:
      print(name + " model failed to execute")
      print(exception)
  scores = {
      "Model": names,
      "Adjusted R2": ADJR2,
      "Mean Absolute Error": MAE,
      "Mean Squared Error": MSE,
      "Root Mean Squared Error": RMSE
  }
  scores = pd.DataFrame(scores)
  if(metric == 'Adjusted R2'):
    scores = scores.sort_values(by='Adjusted R2', ascending=False)
  else:
    scores = scores.sort_values(by=metric, ascending=True)
  TOP3 = [x for x in scores["Model"].head(3)]
  scores = scores.set_index("Model")
  for i in TOP3:
    if i == 'ard_regression':
        ard_reg_hyperopt(metric)
    elif i == 'adaboost':
        adaboost_reg_hyperopt(metric)
    elif i == 'decision_tree':
        decisionTree_reg_hyperopt(metric)
    elif i == 'extra_trees':
        extraTree_reg_hyperopt(metric)
    elif i == 'gaussian_process':
        gaussianProcess_reg_hyperopt(metric)
    elif i == 'gradient_boosting':
        histGradient_reg_hyperopt(metric)
    elif i == 'k_nearest_neighbors':
        kneighbor_reg_hyperopt(metric)
    elif i == 'libsvm_svr':
        svr_reg_hyperopt(metric)
    elif i == 'mlp':
        mlp_reg_hyperopt(metric)
    elif i == 'random_forest':
        randomforest_reg_hyperopt(metric)





#Parameter Tuning
def common_imports():
  from sklearn.model_selection import train_test_split
  X = df.drop([col_names[targetVariable]], axis=1)
  Y = df[col_names[targetVariable]]
  X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)  
  return X_train,y_train
def result_metric(model,X,y):
    from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
    y_pred = model.predict(X)
    precision= precision_score(y,y_pred, average='weighted')
    recall= recall_score(y,y_pred, average='weighted')
    f1_score= f1_score(y,y_pred, average='weighted')
    accuracy= accuracy_score(y,y_pred)
    print(precision, recall, f1_score, accuracy)
def result_metricReg(model, X, y):
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
    print(adj_rsquared, mae, mse, rmse)




def rfc_cls_hyperopt(metric='F1 Score'):
  from hyperopt import hp, fmin, tpe
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import cross_val_score
  X_train, y_train = common_imports()

  def objective(params):
    params = {'criterion': params['criterion'],
              'max_depth': params['max_depth'],
              'max_features': params['max_features'],
              'min_samples_leaf': params['min_samples_leaf'],
              'min_samples_split': params['min_samples_split'],
              'n_estimators': int(params['n_estimators'])
              }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train,  scoring=choice(
        metric), n_jobs=-1, error_score='raise').mean()

    # We aim to maximize f1 score, therefore we return it as a negative value
    return -score
  space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
           'max_depth': hp.quniform('max_depth', 10, 200, 20),
           'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
           'min_samples_leaf': hp.quniform('min_samples_leaf', 0.1, 0.5, 0.1),
           'min_samples_split': hp.quniform('min_samples_split', 0.1, 1, 0.1),
           'n_estimators': hp.quniform('n_estimators', 10, 100, 10)
           }

  best = fmin(fn=objective,
              space=space,
              algo=tpe.suggest,
              max_evals=10)
  max_features = {0: 'auto', 1: 'sqrt', 2: 'log2'}
  best['max_features'] = max_features[best['max_features']]
  criterion = {0: 'entropy', 1: 'gini'}
  best['criterion'] = criterion[best['criterion']]
  model = RandomForestClassifier(n_estimators=int(best['n_estimators']), max_depth=best['max_depth'], min_samples_leaf=best['min_samples_leaf'], max_features=best['max_features'],
                                 criterion=best['criterion'], min_samples_split=best['min_samples_split'])
  model.fit(X_train, y_train)
  result_metric(model, X_train, y_train)
  return model
def adaboost_cls_hyperopt(metric='F1 Score'):
  from hyperopt import fmin, tpe, hp
  from sklearn.ensemble import AdaBoostClassifier
  from sklearn.model_selection import cross_val_score
  X_train, y_train = common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']),
              'learning_rate': params['learning_rate']
              }

    # we use this params to create a new LGBM Regressor
    model = AdaBoostClassifier(**params)

    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choice(
        metric), n_jobs=-1, error_score='raise').mean()
    return -score
  # possible values of parameters

  space = {'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
           'learning_rate': hp.loguniform('learning_rate', -5, 5)
           }

  best = fmin(fn=objective,  # function to optimize
              space=space,
              # optimization algorithm, hyperotp will select its parameters automatically
              algo=tpe.suggest,
              max_evals=10,  # maximum number of iterations
              )
  model = AdaBoostClassifier(n_estimators=int(
      best['n_estimators']), learning_rate=best['learning_rate'])
  model.fit(X_train, y_train)
  result_metric(model, X_train, y_train)
  return model
def decisionTree_cls_hyperopt(metric='F1 Score'):
  from hyperopt import fmin, tpe, hp
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import cross_val_score
  X_train, y_train = common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'splitter': params['splitter'],
              'max_depth': params['max_depth'],
              'min_samples_split': params['min_samples_split'],
              'min_samples_leaf': params['min_samples_leaf'],
              'max_features': params['max_features']
              }
    # we use this params to create a new LGBM Regressor
    model = DecisionTreeClassifier(**params)

    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choice(
        metric), n_jobs=-1, error_score='raise').mean()

    return -score

  # possible values of parameters
  space = {'splitter': hp.choice('splitter', ['best', 'random']),
           'max_depth': hp.quniform('max_depth', 2, 32, 2),
           'min_samples_split': hp.quniform('min_samples_split', 0.1, 1, 0.1),
           'min_samples_leaf': hp.quniform('min_samples_leaf', 0.1, 0.5, 0.1),
           'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])
           }

  best = fmin(fn=objective,  # function to optimize
              space=space,
              # optimization algorithm, hyperotp will select its parameters automatically
              algo=tpe.suggest,
              max_evals=10,  # maximum number of iterations
              )
  max_features = {0: 'auto', 1: 'sqrt', 2: 'log2'}
  splitter = {0: 'best', 1: 'random'}
  best['max_features'] = max_features[best['max_features']]
  best['splitter'] = splitter[best['splitter']]
  model = DecisionTreeClassifier(max_depth=best['max_depth'], min_samples_split=best['min_samples_split'], min_samples_leaf=best['min_samples_leaf'], max_features=best['max_features'],
                                 splitter=best['splitter'])
  model.fit(X_train, y_train)
  result_metric(model, X_train, y_train)
  return model
def extraTree_cls_hyperopt(metric='F1 Score'):
  from hyperopt import fmin, tpe, hp
  from sklearn.ensemble import ExtraTreesClassifier
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators':int(params['n_estimators']),
              'max_depth': params['max_depth'], 
              'min_samples_split': params['min_samples_split'], 
              'min_samples_leaf': params['min_samples_leaf'],
              'max_features': params['max_features']
              }
    # we use this params to create a new LGBM Regressor
    model = ExtraTreesClassifier(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choice(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  
  # possible values of parameters
  space={ 'n_estimators' : hp.quniform('n_estimators',5,100,10),
         'max_depth': hp.quniform('max_depth', 2, 32,2),
         'min_samples_split': hp.quniform('min_samples_split',0.1,1,0.1),
         'min_samples_leaf': hp.quniform('min_samples_leaf',0.1,0.5,0.1),
         'max_features': hp.choice('max_features', ['auto','sqrt','log2'])
        }

  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  max_features={ 0: 'auto', 1: 'sqrt',2:'log2' }
  best['max_features']=max_features[best['max_features']]
  model=ExtraTreesClassifier(n_estimators=int(best['n_estimators']),max_depth=best['max_depth'],min_samples_split=best['min_samples_split'],
                            min_samples_leaf=best['min_samples_leaf'],max_features=best['max_features'])
                      
  model.fit(X_train,y_train)
  result_metric(model,X_train,y_train)
  return model
def gaussianb_cls_hyperopt(metric='F1 Score'):
  from hyperopt import fmin, tpe, hp
  from sklearn.naive_bayes import GaussianNB
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'var_smoothing':params['var_smoothing']
              } 
    # we use this params to create a new LGBM Regressor
    model = GaussianNB(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choice(metric), n_jobs=-1,error_score='raise').mean()
    return -score
 
  # possible values of parameters
  space={ 'var_smoothing' : hp.loguniform('var_smoothing',-10,10)
        }
  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  model=GaussianNB(var_smoothing=best['var_smoothing'])
                      
  model.fit(X_train,y_train)
  result_metric(model,X_train,y_train)
  return model
def histGradient_cls_hyperopt(metric='F1 Score'):
  from hyperopt import fmin, tpe, hp
  from sklearn.ensemble import HistGradientBoostingClassifier
  from sklearn.model_selection import cross_val_score
  X_train, y_train = common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {
        'max_depth': params['max_depth'],
        'max_bins': int(params['max_bins']),
        'max_iter': int(params['max_iter']),
        'max_leaf_nodes': params['max_leaf_nodes'],
        'learning_rate': params['learning_rate']
    }

    # we use this params to create a new LGBM Regressor
    model = HistGradientBoostingClassifier(**params)

    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choice(
        metric), n_jobs=-1, error_score='raise').mean()
    return -score
  # possible values of parameters

  space = {
      'max_depth': hp.quniform('max_depth', 5, 100, 10),
      'max_bins': hp.quniform('max_bins', 5, 250, 10),
      'max_iter': hp.quniform('max_iter', 5, 100, 10),
      'max_leaf_nodes': hp.quniform('max_leaf_nodes', 5, 100, 10),
      'learning_rate': hp.loguniform('learning_rate', -5, 5)
  }

  best = fmin(fn=objective,  # function to optimize
              space=space,
              # optimization algorithm, hyperotp will select its parameters automatically
              algo=tpe.suggest,
              max_evals=10,  # maximum number of iterations
              )
  model = HistGradientBoostingClassifier(max_depth=best['max_depth'], max_bins=int(best['max_bins']), learning_rate=best['learning_rate'],
                                         max_iter=int(best['max_iter']), max_leaf_nodes=best['max_leaf_nodes'])
  model.fit(X_train, y_train)
  result_metric(model, X_train, y_train)
  return model
def kneighbor_cls_hyperopt(metric='F1 Score'):
  from hyperopt import fmin, tpe, hp
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()
  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'n_neighbors':int(params['n_neighbors']),
                'leaf_size':params['leaf_size'],
                'p':params['p'],
                'weights':params['weights'],
               'algorithm':params['algorithm']
              }
    
    # we use this params to create a new LGBM Regressor
    model = KNeighborsClassifier(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choice(metric), n_jobs=-1,error_score='raise').mean()
    return -score
 
  # possible values of parameters
  space={ 'n_neighbors' : hp.quniform('n_neighbors',5,80,10),
         'leaf_size':hp.quniform('leaf_size',5,100,10),
         'p':hp.quniform('p',1,2,1),
          'weights': hp.choice('weights', ['uniform','distance']),
          'algorithm':hp.choice('algorithm',['auto','ball_tree','kd_tree'])
        }

  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  algorithm={ 0: 'auto', 1: 'ball_tree',2:'kd_tree' }
  best['algorithm']=algorithm[best['algorithm']]
  weights={ 0: 'uniform', 1: 'distance' }
  best['weights']=weights[best['weights']]
  model=KNeighborsClassifier(n_neighbors=int(best['n_neighbors']),leaf_size=best['leaf_size'],p=best['p'],weights=best['weights'],algorithm=best['algorithm'])  
  model.fit(X_train,y_train)
  result_metric(model,X_train,y_train)
  return model
def svr_cls_hyperopt(metric='F1 Score'):
  from hyperopt import fmin, tpe, hp
  from sklearn.svm import SVC
  from sklearn.model_selection import cross_val_score
  X_train, y_train = common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'tol': params['tol'],
              'max_iter': params['max_iter'],
              'kernel': params['kernel'],
              'gamma': params['gamma']
              }

    # we use this params to create a new LGBM Regressor
    model = SVC(**params)

    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,
                            scoring=choice(metric), n_jobs=-1).mean()
    return -score
  # possible values of parameters

  space = {'tol': hp.loguniform('tol', -5, 5),
           'max_iter': hp.quniform('max_iter', 5, 100, 10),
           'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
           'gamma': hp.choice('gamma', ['scale', 'auto'])
           }
  best = fmin(fn=objective,  # function to optimize
              space=space,
              # optimization algorithm, hyperotp will select its parameters automatically
              algo=tpe.suggest,
              max_evals=10,  # maximum number of iterations
              )
  kernel = {0: 'linear', 1: 'poly', 2: 'rbf', 3: 'sigmoid'}
  best['kernel'] = kernel[best['kernel']]
  gamma = {0: 'scale', 1: 'auto'}
  best['gamma'] = gamma[best['gamma']]
  model = SVC(tol=best['tol'], max_iter=best['max_iter'],
              kernel=best['kernel'], gamma=best['gamma'])
  model.fit(X_train, y_train)
  result_metric(model, X_train, y_train)
  return model
def mlp_cls_hyperopt(metric='F1 Score'):
  from hyperopt import fmin, tpe, hp
  from sklearn.neural_network import MLPClassifier
  from sklearn.model_selection import cross_val_score
  X_train, y_train = common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'alpha': params['alpha'],
              'max_iter': int(params['max_iter']),
              'activation': params['activation']
              }

    # we use this params to create a new LGBM Regressor
    model = MLPClassifier(**params)

    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choice(
        metric), n_jobs=-1, error_score='raise').mean()
    return -score
  # possible values of parameters

  space = {'alpha': hp.loguniform('alpha', -5, 5),
           'max_iter': hp.quniform('max_iter', 500, 3000, 200),
           'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu'])
           }
  best = fmin(fn=objective,  # function to optimize
              space=space,
              # optimization algorithm, hyperotp will select its parameters automatically
              algo=tpe.suggest,
              max_evals=10,  # maximum number of iterations
              )
  activation = {0: 'identity', 1: 'logistic', 2: 'tanh', 3: 'relu'}
  best['activation'] = activation[best['activation']]
  model = MLPClassifier(alpha=best['alpha'], max_iter=int(
      best['max_iter']), activation=best['activation'])
  model.fit(X_train, y_train)
  result_metric(model, X_train, y_train)
  return model

def ard_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.linear_model import ARDRegression
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()
  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'n_iter': int(params['n_iter']),
              'alpha_1': params['alpha_1'], 
              'alpha_2': params['alpha_2'], 
              'lambda_1': params['lambda_1'], 
              'lambda_2': params['lambda_2']
    }
    
    # we use this params to create a new LGBM Regressor
    model = ARDRegression(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric),error_score='raise', n_jobs=-1).mean()
    return -score
  # possible values of parameters
  space={'n_iter': hp.quniform('n_iter', 5, 2000, 200),
       'alpha_1': hp.loguniform('alpha_1', -5, 5),
       'alpha_2': hp.loguniform('alpha_2', -5, 5),
       'lambda_1': hp.loguniform('lambda_1', -5, 5),
       'lambda_2': hp.loguniform('lambda_2', -5, 5)
      }

  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  model=ARDRegression(n_iter=int(best['n_iter']),alpha_1=best['alpha_1'],alpha_2=best['alpha_2'],lambda_1=best['lambda_1'],lambda_2=best['lambda_2'])
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
def adaboost_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.ensemble import AdaBoostRegressor
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()
  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']), 
              'learning_rate': params['learning_rate'], 
              'loss': params['loss']
              }
  
    # we use this params to create a new LGBM Regressor
    model = AdaBoostRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters

  space={'n_estimators': hp.quniform('n_estimators', 100,1000,100),
         'learning_rate': hp.loguniform('learning_rate', -5, 5),
         'loss': hp.choice('loss', ['linear','square','exponential'])
        }

  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  loss={ 0: 'linear', 1: 'square',2:'exponential' }
  best['loss']=loss[best['loss']]
  model=AdaBoostRegressor(n_estimators=int(best['n_estimators']),learning_rate=best['learning_rate'],loss=best['loss'])
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
def decisionTree_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'max_depth': params['max_depth'], 
              'min_samples_split': params['min_samples_split'], 
              'min_samples_leaf': params['min_samples_leaf'],
              'max_features': params['max_features'],
              'criterion': params['criterion']
              }
    
    # we use this params to create a new LGBM Regressor
    model = DecisionTreeRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters


  space={'max_depth': hp.quniform('max_depth', 2, 32,2),
         'min_samples_split': hp.quniform('min_samples_split',0.1,1,0.1),
         'min_samples_leaf': hp.quniform('min_samples_leaf',0.1,0.5,0.1),
         'max_features': hp.choice('max_features', ['auto','sqrt','log2']),
         'criterion': hp.choice('criterion', ['squared_error','friedman_mse','absolute_error','poisson'])
        }

  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  max_features={ 0: 'auto', 1: 'sqrt',2:'log2' }
  best['max_features']=max_features[best['max_features']]
  criterion={ 0: 'squared_error', 1: 'friedman_mse',2:'absolute_error',3:'poisson' }
  best['criterion']=criterion[best['criterion']]
  model=DecisionTreeRegressor(max_depth=best['max_depth'],min_samples_split=best['min_samples_split'],min_samples_leaf=best['min_samples_leaf'],max_features=best['max_features'],
                              criterion=best['criterion'])
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
def extraTree_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.ensemble import ExtraTreesRegressor
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators':int(params['n_estimators']),
              'min_samples_split': params['min_samples_split'], 
              'min_samples_leaf': params['min_samples_leaf'],
              'max_features': params['max_features'],
              'criterion': params['criterion']
              }
    
    # we use this params to create a new LGBM Regressor
    model = ExtraTreesRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters

  space={ 'n_estimators' : hp.quniform('n_estimators',5,100,10),
         'min_samples_split': hp.quniform('min_samples_split',0.1,1,0.1),
         'min_samples_leaf': hp.quniform('min_samples_leaf',0.1,0.5,0.1),
         'max_features': hp.choice('max_features', ['auto','sqrt','log2']),
         'criterion': hp.choice('criterion', ['squared_error','absolute_error'])
        }

  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  max_features={ 0: 'auto', 1: 'sqrt',2:'log2' }
  best['max_features']=max_features[best['max_features']]
  criterion={ 0: 'squared_error', 1: 'friedman_mse',2:'absolute_error',3:'poisson' }
  best['criterion']=criterion[best['criterion']]
  model=ExtraTreesRegressor(n_estimators=int(best['n_estimators']),min_samples_split=best['min_samples_split'],min_samples_leaf=best['min_samples_leaf'],max_features=best['max_features'],
                              criterion=best['criterion'])  
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
def gaussianProcess_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'alpha':params['alpha'],
              'n_restarts_optimizer':params['n_restarts_optimizer']
              }
    
    # we use this params to create a new LGBM Regressor
    model = GaussianProcessRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters

  space={ 'alpha' : hp.loguniform('alpha',-5,5),
         'n_restarts_optimizer': hp.quniform('n_restarts_optimizer', 0,2,1)
                 }
  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  model=GaussianProcessRegressor(alpha=best['alpha'],n_restarts_optimizer=best['n_restarts_optimizer'])  
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
def histGradient_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.ensemble import HistGradientBoostingRegressor
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {
              'max_depth': params['max_depth'], 
              'max_bins':int(params['max_bins']),
              'max_iter':int(params['max_iter']),
              'max_leaf_nodes':params['max_leaf_nodes'],
              'learning_rate':params['learning_rate'],
              'loss':params['loss']
              }
    
    # we use this params to create a new LGBM Regressor
    model = HistGradientBoostingRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters
  space={
         'max_depth': hp.quniform('max_depth', 5,100,10),
         'max_bins':hp.quniform('max_bins',5,250,10),
          'max_iter': hp.quniform('max_iter', 5,100,10),
         'max_leaf_nodes':hp.quniform('max_leaf_nodes',5,100,10),
          'learning_rate': hp.loguniform('learning_rate', -5, 5),
         'loss':hp.choice('loss', ['squared_error', 'least_squares', 'absolute_error', 'least_absolute_deviation'])
        }
  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  loss={ 0: 'squared_error', 1: 'least_squares',2:'absolute_error',3:'least_absolute_deviation' }
  best['loss']=loss[best['loss']]
  model=HistGradientBoostingRegressor(max_depth=best['max_depth'],max_bins=int(best['max_bins']),learning_rate=best['learning_rate'],
                                      max_iter=int(best['max_iter']),max_leaf_nodes=best['max_leaf_nodes'],loss=best['loss']) 
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model 
def kneighbor_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()
  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'n_neighbors':int(params['n_neighbors']),
                'leaf_size':params['leaf_size'],
                'p':params['p'],
                'weights':params['weights'],
                'algorithm':params['algorithm']
              }
    
    # we use this params to create a new LGBM Regressor
    model = KNeighborsRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters

  space={ 'n_neighbors' : hp.quniform('n_neighbors',5,100,10),
         'leaf_size':hp.quniform('leaf_size',5,100,10),
         'p':hp.quniform('p',1,2,1),
          'weights': hp.choice('weights', ['uniform','distance']),
          'algorithm':hp.choice('algorithm',['auto','ball_tree','kd_tree'])
        }

  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  weights={ 0: 'uniform', 1: 'distance' }
  best['weights']=weights[best['weights']]
  algorithm={ 0: 'auto', 1: 'ball_tree',2:'kd_tree' }
  best['algorithm']=algorithm[best['algorithm']]
  model=KNeighborsRegressor(n_neighbors=int(best['n_neighbors']),leaf_size=best['leaf_size'],p=best['p'],weights=best['weights'],algorithm=best['algorithm'])  
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
def svr_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.svm import SVR
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'tol':params['tol'],
              'max_iter':params['max_iter'],
              'kernel':params['kernel'],
              'gamma':params['gamma']
              }
    
    # we use this params to create a new LGBM Regressor
    model = SVR(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters

  space={ 'tol' : hp.loguniform('tol',-5,5),
         'max_iter':hp.quniform('max_iter',5,100,10),
         'kernel': hp.choice('kernel', ['linear','poly','rbf','sigmoid']),
         'gamma':hp.choice('gamma',['scale','auto'])
        }
  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  kernel={ 0: 'kernel', 1: 'linear',2:'poly',3:'rbf',4:'sigmoid' }
  best['kernel']=kernel[best['kernel']]
  gamma={ 0: 'scale', 1: 'auto' }
  best['gamma']=gamma[best['gamma']]
  model=SVR(tol=best['tol'],max_iter=best['max_iter'],kernel=best['kernel'],gamma=best['gamma'])  
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
def mlp_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.neural_network import MLPRegressor
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()

  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'alpha':params['alpha'],
              'max_iter':int(params['max_iter']),
              'activation':params['activation']
              }
    
    # we use this params to create a new LGBM Regressor
    model = MLPRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters

  space={ 'alpha' : hp.loguniform('alpha',-5,5),
         'max_iter':hp.quniform('max_iter',500,3000,200),
         'activation': hp.choice('activation', [ 'identity', 'logistic', 'tanh', 'relu'])
        }
  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  activation={ 0: 'identity', 1: 'logistic',2:'tanh',3:'relu' }
  best['activation']=activation[best['activation']]
  model=MLPRegressor(alpha=best['alpha'],max_iter=int(best['max_iter']),activation=best['activation'])  
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
def randomforest_reg_hyperopt(metric='Adjusted R2'):
  from hyperopt import fmin, tpe, hp
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import cross_val_score
  X_train,y_train=common_imports()
  def objective(params):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators':int(params['n_estimators']),
              'min_samples_leaf':params['min_samples_leaf'],
              'max_features':params['max_features'],
              'max_leaf_nodes':int(params['max_leaf_nodes'])
              }
    
    # we use this params to create a new LGBM Regressor
    model = RandomForestRegressor(**params)
    
    # and then conduct the cross validation with the same folds as before
    score = cross_val_score(model, X_train, y_train,  scoring=choiceReg(metric), n_jobs=-1,error_score='raise').mean()
    return -score
  # possible values of parameters

  space={ 'n_estimators' : hp.quniform('n_estimators',10,100,10),
         'min_samples_leaf': hp.quniform('min_samples_leaf',0.1,0.5,0.1),
         'max_features': hp.choice('max_features', ['auto','sqrt','log2']),
         'max_leaf_nodes':hp.quniform('max_leaf_nodes',10,50,10)
        }
  best=fmin(fn=objective, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
         )
  max_features={ 0: 'auto', 1: 'sqrt',2:'log2' }
  best['max_features']=max_features[best['max_features']]
  model=RandomForestRegressor(n_estimators=int(best['n_estimators']),min_samples_leaf=best['min_samples_leaf'],max_features=best['max_features'],
                     max_leaf_nodes=int(best['max_leaf_nodes']))  
  model.fit(X_train,y_train)
  result_metricReg(model,X_train,y_train)
  return model
