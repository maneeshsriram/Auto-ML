from sklearn.linear_model import LinearRegression
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
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








def modelmaking(request):
    return render(request, 'webpages/model.html')





def allModels(request):
    random_state = 42
    prediction = False
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
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )  
    ADJR2 = []
    predictions = {}
    names = []
    TIME = []
    CUSTOM_METRIC = []
    from sklearn.datasets import load_diabetes
    db=load_diabetes()
    X=db.data
    y=db.target
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)  
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, numeric_features),("categorical", categorical_transformer, categorical_features),])
    for name, model in REGRESSORS:
        try:
            if "random_state" in model().get_params().keys():
                pipe = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("regressor", model(random_state=random_state))
                        ]
                        )
            else:
                pipe = Pipeline(
                    steps=[("preprocessor", preprocessor), ("regressor", model())]
                    )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            r_squared = r2_score(y_test, y_pred)
            n = X_test.shape[0]
            p = X_test.shape[1]
            adj_rsquared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))
            names.append(name)
            ADJR2.append(adj_rsquared)

            if prediction:
                predictions[name] = y_pred
        except Exception as exception:
            print(name + " model failed to execute")
            print(exception)
    scores = {
            "Model": names,
            "Adjusted R-Squared": ADJR2,
            }
    scores = pd.DataFrame(scores)
        
    scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index("Model")
        
    if prediction:
        predictions_df = pd.DataFrame.from_dict(predictions)
        return predictions
    
    scores_dict = scores.to_dict()

    data = {
        'ard_regression': scores_dict['Adjusted R-Squared']['ard_regression'],
        'adaboost': scores_dict['Adjusted R-Squared']['adaboost'],
        'extra_trees': scores_dict['Adjusted R-Squared']['extra_trees'],
        'random_forest': scores_dict['Adjusted R-Squared']['random_forest'],
        'gradient_boosting': scores_dict['Adjusted R-Squared']['gradient_boosting'],
        'k_nearest_neighbors': scores_dict['Adjusted R-Squared']['k_nearest_neighbors'],
        'libsvm_svr': scores_dict['Adjusted R-Squared']['libsvm_svr'],
        'decision_tree': scores_dict['Adjusted R-Squared']['decision_tree'],
        'gaussian_process': scores_dict['Adjusted R-Squared']['gaussian_process'],
        'mlp': scores_dict['Adjusted R-Squared']['mlp'],
    }
    return render(request, 'webpages/results/resAllModels.html', data)

    


def modelList(request):
    return render(request, 'webpages/modelsList.html')

def linear(request):
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = LinearRegression()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def ridge(request):
    from sklearn.linear_model import Ridge
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split,cross_val_score
    X,Y = load_diabetes(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0) 
    model=Ridge()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n=X_train.shape[0]
    p=X_train.shape[1]
    adj_rsquared=1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def lasso(request):
    from sklearn.linear_model import Lasso
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = Lasso()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def enr(request):
    from sklearn.linear_model import ElasticNet
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = ElasticNet()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def ard(request):
    from sklearn.linear_model import ARDRegression
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    import pandas as pd
    import numpy as np
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    model = ARDRegression()
    r_squared = cross_val_score(
        model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def sgd(request):
    from sklearn.linear_model import SGDRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = SGDRegressor()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def svr(request):
    from sklearn.svm import SVR
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = SVR()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def dtr(request):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = DecisionTreeRegressor()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def rfr(request):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = RandomForestRegressor()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def gbr(request):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = GradientBoostingRegressor()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def lgbm(request):
    from lightgbm import LGBMRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split,cross_val_score
    X,Y = load_diabetes(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)    
    model=LGBMRegressor()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean() 
    n=X_train.shape[0]
    p=X_train.shape[1]
    adj_rsquared=1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def xgbr(request):
    from xgboost.sklearn import XGBRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split,cross_val_score
    X,Y = load_diabetes(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)    
    model=XGBRegressor(verbosity=0)
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()                         
    n=X_train.shape[0]
    p=X_train.shape[1]
    adj_rsquared=1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def guassian(request):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = GaussianProcessRegressor()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def knr(request):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = KNeighborsRegressor()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)

def mlp(request):
    from sklearn.neural_network import MLPRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, cross_val_score
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    model = MLPRegressor()
    r_squared = cross_val_score(model, X_train, y_train,  scoring="r2", cv=5).mean()
    n = X_train.shape[0]
    p = X_train.shape[1]
    adj_rsquared = 1-(1-r_squared)*((n-1)/(n-p-1))
    data = {
        'score': round(adj_rsquared, 2)
    }
    return render(request, 'webpages/results/resListModels.html', data)


