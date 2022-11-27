from turtle import title
from django.shortcuts import render

import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


def home(request):
    return render(request, 'webpages/home.html')


def overview(request):
    return render(request, 'webpages/overview.html')


def resOverview(request):
    if request.method == 'POST':
        file = request.FILES['csvfile']
        numericaldata = request.POST['num']

        df = pd.read_csv(file)

        numerical = []
        for i in numericaldata.split():
            numerical.append(df.columns[int(i)])
        
        col_name = set(df.columns.tolist())
        num = set(numerical)
        categorical = col_name - num
        categorical = list(categorical)

        numericalValues = []
        categoricalValues = []

        for i in numerical:  # DATA OF ALL NUMERICAL COLUMNS
            objNum = {
                'featureName': (i),
                'CountOfNonEmptyValues': (df[i].count()),
                'CountOfEmptyValues': (df[i].isnull().sum()),
                'PercentageOfEmptyValues': (df[i].isnull().sum()*100 /(df.shape[0])),
                'mean': (df[i].mean()),
                'median': (df[i].median()),
                'mode': (df[i].mode()[0]),
                'variance': (df[i].std()*df[i].std()),
                'standardDeviation': (df[i].std()),
                'minimumValue': (df[i].min()),
                'maximumValue': (df[i].max()),
                'q1': (df[i].quantile(0.25)),
                'q2': (df[i].quantile(0.5)),
                'q3': (df[i].quantile(0.75)),
                'iqr': (df[i].quantile(0.75)-df[i].quantile(0.25)),
                'kurtosis': (df[i].kurtosis())
            }
            numericalValues.append(objNum)

        for i in categorical:  # DATA OF ALL CATEGORICAL COLUMNS
            objCat = {
                'featureName': (i),
                'CountOfNonEmptyValues': (df[i].count()),
                'CountOfEmptyValues': (df[i].isnull().sum()),
                'PercentageOfEmptyValues': (df[i].isnull().sum()*100 / (df[i].count()+df[i].isnull().sum())),
                'mode': (df[i].mode()[0])
            }
            categoricalValues.append(objCat)

        data = {
            'numberOfColumns': df.shape[1],
            'numberOfRows': df.shape[0],
            'numberOfDuplicate': df[df.duplicated()].count()[0],
            'numberOfNumericalColumns': len(numerical),
            'numberOfCategoricalColumns': len(categorical),
            'numericalValues': numericalValues,
            'categoricalValues': categoricalValues,
        }
        print(numericalValues)
    return render(request, 'webpages/results/resOverview.html', data)






def visualization(request):
    return render(request, 'webpages/visualization.html')

def resVisualization(request):
    if request.method == 'POST':
        file = request.FILES['csvfile']
        numericaldata = request.POST['num']
        df = pd.read_csv(file)

        numerical = []
        for i in numericaldata.split():
            numerical.append(df.columns[int(i)])
        
        col_name = set(df.columns.tolist())
        num = set(numerical)
        categorical = col_name - num
        categorical = list(categorical)

        #Heatmap
        fig = px.imshow(df.corr(), text_auto=True, title="Heatmap")
        fig.update_layout(title={'font_size':22, 'xanchor': 'center', 'x':0.5})
        heat = fig.to_html()

        #Histogram
        histo = []
        for i in numerical:
            try:
                fig = px.histogram(df[i], nbins=11, labels={'value': i}, title="Histogram")
                fig.update_layout(bargap=0.2, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), showlegend=False, title={'font_size':22, 'xanchor': 'center', 'x':0.5})
                histo.append(fig.to_html())
            except:
                continue

        #Pie plot
        pie = []
        for i in categorical:
            try:
                if(df[i].nunique() < 20):
                    fig = px.pie(df[i].value_counts(), names=df[i].unique(), title="Pie Plot") 
                    fig.update_layout(title={'font_size':22, 'xanchor': 'center', 'x':0.5})
                    pie.append(fig.to_html())
            except:
                continue

        #Count plot
        count = []
        for i in categorical:
            try:
                if(df[i].nunique() < 0.1*len(df)):
                    fig = px.bar(df[i], title="Count Plot")
                    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), showlegend=False, title={'font_size': 22, 'xanchor': 'center', 'x': 0.5})
                    count.append(fig.to_html())
            except:
                continue

        #Box plot
        box = []
        for i in numerical:
            try:
                fig = px.box(df[i], title="Box Plot")
                fig.update_layout(title={'font_size': 22, 'xanchor': 'center', 'x': 0.5})
                box.append(fig.to_html())
            except:
                continue

        #Line plot
        line = []
        for i in numerical:
            for j in numerical:
                try:
                    if(i != j):
                        fig = px.line(df.sort_values(by=i), x=i, y=j, labels={'x': i, 'y': i}, title="Line Plot")
                        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), showlegend=False, title={'font_size': 22, 'xanchor': 'center', 'x': 0.5})
                        line.append(fig.to_html())
                except:
                    continue

        #Scatter plot
        scatter = []
        for i in numerical:
            for j in numerical:
                try:
                    if(i != j):
                        fig = px.scatter(df.sort_values(by=i), x=i, y=j, labels={'x': i, 'y': j}, title='Scatter plot')
                        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), showlegend=False, title={'font_size': 22, 'xanchor': 'center', 'x': 0.5})
                        scatter.append(fig.to_html())
                except:
                    continue

        data = {
            'heatmap': heat,
            'histogram': histo,
            'pie' : pie,
            'count' : count,
            'box' : box,
            'line' : line,
            'scatter': scatter
        }
    return render(request, 'webpages/results/resVisualization.html', data)




def parameter(request):
    return render(request, 'webpages/parameter.html')

def deployment(request):
    return render(request, 'webpages/deployment.html')




def predictionDataset(request):
    return render(request, 'webpages/predictionDataset.html')

def prediction(request):
    if request.POST['model'] == 'clas':
        model = joblib.load(request.FILES['pklfile'])
        df = pd.read_csv(request.FILES['csvfile'])
        col_name = (df.columns.tolist())
        Y = col_name[int(request.POST['tar'])]
        col_name.remove(Y)
        X = col_name
        X = df.drop(Y, axis=1)
        Y = df[Y]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
        precision = cross_val_score(model, x_test, y_test,scoring='precision_weighted',error_score='raise').mean()
        recall = cross_val_score(model, x_test, y_test,scoring='recall_weighted').mean()
        f1 = cross_val_score(model, x_test, y_test,scoring='f1_weighted').mean()
        accuracy = cross_val_score(model, x_test, y_test,scoring='accuracy').mean()
        data = {
            "pre": precision,
            "rec": recall,
            "f1_score": f1,
            "acc":accuracy,
            "model": model
        }
        return render(request, 'webpages/results/resListClassModels.html', data)
    else:
            model = joblib.load(request.FILES['pklfile'])
            df = pd.read_csv(request.FILES['csvfile'])
            col_name = (df.columns.tolist())
            Y = col_name[int(request.POST['tar'])]
            col_name.remove(Y)
            X = col_name
            X = df.drop(Y, axis=1)
            Y = df[Y]
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
            r_squared = cross_val_score(model, x_test, y_test,  scoring="r2", cv=5).mean()
            n=x_test.shape[0]
            p=x_test.shape[1]
            adj_rsquared=1-(1-r_squared)*((n-1)/(n-p-1))
            mae = (-1) *cross_val_score(model, x_test, y_test,  scoring="neg_mean_absolute_error").mean()
            mse = (-1) *cross_val_score(model, x_test, y_test,  scoring="neg_mean_squared_error").mean()
            rmse= (-1) *cross_val_score(model, x_test, y_test,  scoring="neg_root_mean_squared_error").mean()
            data = {
                "adj_rsquared": adj_rsquared,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "model": model
            }
            return render(request, 'webpages/results/resListRegModels.html', data)


    

    
    











