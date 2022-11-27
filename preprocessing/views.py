from django.shortcuts import render
import pandas as pd
import numpy as np
import random
from django.http import HttpResponse
import csv

# Create your views here.


def preprocessingForm(request):
    return render(request, 'webpages/preprocessing.html')


def preprocessingCol(request):
    if request.method == 'POST':
        global numericaldata
        global df
        global col_name
        numericaldata = (request.POST['num']).split()
        file = request.FILES['csvfile']
        df = pd.read_csv(file)
        col_name = df.columns.tolist()
    return render(request, 'webpages/preprocessingCol.html')




def resPreprocessing(request):
    methods = (request.POST['met']).split()
    for i in range(len(methods)):  
        if(methods[i] == '0'):  # skip preprocessing
            pass
        elif(methods[i] == '1'):  # drop rows
            df = drop_rows(col_name, i)
        elif(methods[i] == '2'):  # drop column
            df = drop_col_explicit(col_name, i)
        elif(methods[i] == '3'):   # impute mean , if categorical use mode(to do this used try and except)
            try:
                df = impute_mean(col_name, i)
            except:
                df = impute_mode(col_name, i)
        elif(methods[i] == '4'):   # impute median , if categorical use mode(to do this used try and except)
            try:
                df = impute_median(col_name, i)
            except:
                df = impute_mode(col_name, i)
        elif(methods[i] == '5'):  # impute mode
            df = impute_mode(col_name, i)
        elif(methods[i] == '6'):  # do forward fill
            df = forward_fill(col_name, i)
        elif(methods[i] == '7'):      # do backward fill
            df = backward_fill(col_name, i)
        elif(methods[i] == '8'):          # fill null with random values from that column
            df = fill_random_values(col_name, i)
    return render(request, 'webpages/results/resPreprocessing.html')

def drop_rows(col_name, i):  # keep rows if atleast half columns are not NA
    df.dropna(thresh=len(col_name)/2, inplace=True)
    return df

def drop_col_explicit(col_name, i):
    df.drop([col_name[i]], axis=1, inplace=True)
    return df

def impute_mean(col_name, i):
    df[col_name[i]].fillna(df[col_name[i]].mean(), inplace=True)
    return df

def impute_median(col_name,i):
    df[col_name[i]].fillna(df[col_name[i]].median(), inplace = True)
    return df

def impute_mode(col_name,i):
    df[col_name[i]].fillna(df[col_name[i]].mode()[0], inplace = True)
    return df

def forward_fill(col_name, i):
    df[col_name[i]].fillna(method ='ffill', inplace = True)
    return df

def backward_fill(col_name, i):
    df[col_name[i]].fillna(method ='bfill', inplace = True)
    return df

def fill_random_values(col_name,i):
    rows_with_nan = [index for index, row in df[col_name[i]].to_frame().iterrows() if row.isnull().any()]
    l=df[col_name[i]][~df[col_name[i]].isna()].tolist()
    for j in rows_with_nan:
        df[col_name[i]][j]=random.choice(l)
    return df

def preprocessingDownload(request):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="Preprocessed.csv"'
        writer = csv.writer(response)
        col_name_upd = df.columns.tolist()
        writer.writerow(col_name_upd)
        rows = df.to_numpy().tolist()
        for i in range(len(rows)):
            writer.writerow(rows[i])
        return response





def preprocessingOutlier(request):
    return render(request, 'webpages/preprocessingOutlier.html')

def resPreprocessingOutlier(request):
    methods = (request.POST['Omet']).split()
    for i in range(len(methods)):
        i = int(i)
        if(methods[i] == '0'):  
            pass
        elif(methods[i] == '1'):  
            df = iqr(i)
    return render(request, 'webpages/results/resPreprocessingOutlier.html')


def iqr(i):
        col_name_outl = []
        for j in numericaldata: 
            j = int(j)
            col_name_outl.append(col_name[j])
        q1 = np.percentile(df[col_name_outl[i]], 25)
        q3 = np.percentile(df[col_name_outl[i]], 75)
        print("IIIIIIIIIIiiiiiiiiii")
        IQR = q3-q1
        lwr_bound = q1-(1.5*IQR)
        upr_bound = q3+(1.5*IQR)
        for j in range(0,df.shape[0]): 
            print("hiiiiiiiiiii")
            if (df[col_name_outl[i]][j]<lwr_bound or df[col_name_outl[i]][j]>upr_bound):
                df.drop(j,inplace=True)
        return df
