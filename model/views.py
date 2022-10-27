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

    
    




def chooseModel(request):
    pass
    
