from django.shortcuts import render

import pandas as pd
import numpy as np
import random



def home(request):
    return render(request, 'webpages/home.html')


def preprocessing(request):
    return render(request, 'webpages/preprocessing.html')
def visualization(request):
    return render(request, 'webpages/visualization.html')
def model(request):
    return render(request, 'webpages/model.html')
def parameter(request):
    return render(request, 'webpages/parameter.html')
def prediction(request):
    return render(request, 'webpages/prediction.html')
def deployment(request):
    return render(request, 'webpages/deployment.html')


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
    return render(request, 'webpages/results/resPreprocessing.html', data)


def resVisualization(request):
    return render(request, 'webpages/results/resVisualization.html')










