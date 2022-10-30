from django.core.files.storage import default_storage
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
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, make_scorer



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







def allModelsRegression(request):
    file = default_storage.open(file_name)
    random_state = 42
    prediction = False
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
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")),
               ("scaler", StandardScaler())]
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
    db = load_diabetes()
    X = db.data
    y = db.target
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, numeric_features), ("categorical", categorical_transformer, categorical_features), ])
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
                    steps=[("preprocessor", preprocessor),
                           ("regressor", model())]
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

    scores = scores.sort_values(
        by="Adjusted R-Squared", ascending=False).set_index("Model")

    if prediction:
        predictions_df = pd.DataFrame.from_dict(predictions)
        return predictions

    r2_scores_dict = scores.to_dict()
    data = {
        'ard_regression': r2_scores_dict['Adjusted R-Squared']['ard_regression'],
        'adaboost': r2_scores_dict['Adjusted R-Squared']['adaboost'],
        'extra_trees': r2_scores_dict['Adjusted R-Squared']['extra_trees'],
        'random_forest': r2_scores_dict['Adjusted R-Squared']['random_forest'],
        'gradient_boosting': r2_scores_dict['Adjusted R-Squared']['gradient_boosting'],
        'k_nearest_neighbors': r2_scores_dict['Adjusted R-Squared']['k_nearest_neighbors'],
        'libsvm_svr': r2_scores_dict['Adjusted R-Squared']['libsvm_svr'],
        'decision_tree': r2_scores_dict['Adjusted R-Squared']['decision_tree'],
        'gaussian_process': r2_scores_dict['Adjusted R-Squared']['gaussian_process'],
        'mlp': r2_scores_dict['Adjusted R-Squared']['mlp'],
    }
    return render(request, 'webpages/results/resAllModelsRegression.html', data)


def allModelsClassification(request):
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
    names = []
    X, Y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    for name, model in CLASSIFIERS:
        try:
            f1 = cross_val_score(model(), X_train, y_train, cv=5, scoring=make_scorer(
                f1_score, average='micro')).mean()
            names.append(name)
            F1_Score.append(f1)
        except Exception as exception:
            print(name + " model failed to execute")
            print(exception)
    f2_scores = {
        "Model": names,
        "F1 Score": F1_Score,
    }
    data = {
        'Cadaboost': f2_scores['F1 Score'][0],
        'Cdecision_tree': f2_scores['F1 Score'][1],
        'Cextra_trees': f2_scores['F1 Score'][2],
        'CgaussianNB': f2_scores['F1 Score'][3],
        'Cgradient_boosting': f2_scores['F1 Score'][4],
        'Ck_nearest_neighbors': f2_scores['F1 Score'][5],
        'Clibsvm_svc': f2_scores['F1 Score'][6],
        'Cmlp': f2_scores['F1 Score'][7],
        'Crandom_forest': f2_scores['F1 Score'][8],
    }
    return render(request, 'webpages/results/resAllModelsClassification.html', data)
    

    






#Regression Models
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


# Classification Models
def logistic(request):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def svc(request):
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=SVC()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def dtc(request):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=DecisionTreeClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def gaussianNB(request):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=GaussianNB()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def multinomialNB(request):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=MultinomialNB()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def sgdc(request):
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=SGDClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def knnc(request):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=KNeighborsClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def rfc(request):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def gbc(request):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=GradientBoostingClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def lgbmc(request):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=LGBMClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)

def xgbc(request):
    from xgboost.sklearn import XGBClassifier
    from sklearn.metrics import f1_score   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)    
    model=XGBClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)   
    f1=f1_score(y_test,y_pred,average='micro')
    data = {
        'score': f1
    }
    return render(request, 'webpages/results/resListModels.html', data)




