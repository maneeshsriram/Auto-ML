from turtle import title
from django.shortcuts import render

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


def home(request):
    return render(request, 'webpages/home.html')





def preprocessing(request):
    return render(request, 'webpages/preprocessing.html')

def resPreprocessing(request):
    if request.method == 'POST':
        file = request.FILES['csvfile']
        numericaldata = request.POST['num']
        targetvariable = request.POST['tar']

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
    return render(request, 'webpages/results/resPreprocessing.html', data)





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


def prediction(request):
    return render(request, 'webpages/prediction.html')


def deployment(request):
    return render(request, 'webpages/deployment.html')








