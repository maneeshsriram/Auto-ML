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
        global col_name_all
        numericaldata = (request.POST['num']).split()
        file = request.FILES['csvfile']
        df = pd.read_csv(file)
        col_name = df.columns.tolist()
        col_name_all = df.columns.tolist()
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
    try:
        df.dropna(thresh=len(col_name)/2, inplace=True)
    except:
        pass
    return df
def drop_col_explicit(col_name, i):
    try:
        df.drop([col_name[i]], axis=1, inplace=True)
    except:
        pass
    return df
def impute_mean(col_name, i):
    try:
        df[col_name[i]].fillna(df[col_name[i]].mean(), inplace=True)
    except:
        pass
    return df
def impute_median(col_name,i):
    try:
        df[col_name[i]].fillna(df[col_name[i]].median(), inplace = True)
    except:
        pass
    return df
def impute_mode(col_name,i):
    try:
        df[col_name[i]].fillna(df[col_name[i]].mode()[0], inplace = True)
    except:
        pass
    return df
def forward_fill(col_name, i):
    try:
        df[col_name[i]].fillna(method ='ffill', inplace = True)
    except:
        pass
    return df
def backward_fill(col_name, i):
    try:
        df[col_name[i]].fillna(method ='bfill', inplace = True)
    except:
        pass
    return df
def fill_random_values(col_name,i):
    try:
        rows_with_nan = [index for index, row in df[col_name[i]].to_frame().iterrows() if row.isnull().any()]
        l=df[col_name[i]][~df[col_name[i]].isna()].tolist()
        for j in rows_with_nan:
            df[col_name[i]][j]=random.choice(l)
    except:
        pass
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
        elif(methods[i] == '2'):
            df = zscore(i)
        elif(methods[i] == '3'):
            df = iso_forest(i)
    return render(request, 'webpages/results/resPreprocessingOutlier.html')
def iqr(i):
        col_name_outl = []
        for j in numericaldata: 
            j = int(j)
            col_name_outl.append(col_name[j])
        q1 = np.percentile(df[col_name_outl[i]], 25)
        q3 = np.percentile(df[col_name_outl[i]], 75)
        IQR = q3-q1
        lwr_bound = q1-(1.5*IQR)
        upr_bound = q3+(1.5*IQR)
        size = df.shape[0]
        for j in range(size): 
            try:
                if (df[col_name_outl[i]][j]<lwr_bound or df[col_name_outl[i]][j]>upr_bound):
                    df.drop(j,inplace=True)
            except:
                pass
        return df
def zscore(i):
    col_name_outl = []
    for j in numericaldata:
        j = int(j)
        col_name_outl.append(col_name[j])
    thres = 3.3
    mean = np.mean(df[col_name_outl[i]])
    std = np.std(df[col_name_outl[i]])
    upr_bound = mean+3.5*std
    lwr_bound = mean-3.5*std
    size = df.shape[0]
    for j in range(0, size):
        try:
            if (df[col_name_outl[i]][j] < lwr_bound or df[col_name_outl[i]][j] > upr_bound):
                df.drop(j, inplace=True)
        except:
            pass
    return df
def iso_forest(i):
    col_name_outl = []
    for j in numericaldata:
        j = int(j)
        col_name_outl.append(col_name[j])
    from sklearn.ensemble import IsolationForest
    model=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.02),random_state=2)
    model.fit(df[[col_name_outl[i]]])
    df['anomaly_score'] = model.predict(df[[col_name_outl[i]]])
    for j in range(0,df.shape[0]):
        try:
            if(df['anomaly_score'][j]==-1):
                df.drop(j,inplace=True)
        except:
            pass
    df.drop(['anomaly_score'],inplace=True, axis = 1)
    return df




def preprocessingFeatureScaling(request):
    return render(request, 'webpages/preprocessingFeatureScaling.html')
def resPreprocessingFeatureScaling(request):
    methods = (request.POST['Fmet']).split()
    global targetVariable
    targetVariable = int(request.POST['tar'])

    global col_name_num
    col_name_num = []
    for j in numericaldata:
        j = int(j)
        col_name_num.append(col_name[j])
    try:
        col_name_num.remove(col_name_all[targetVariable])
    except:
        pass
    for i in range(len(methods)):
        i = int(i)
        if(methods[i] == '0'):
            pass
        elif(methods[i] == '1'):
            df = absolute_maximum_scale(i)
        elif(methods[i] == '2'):
            df = min_max_scale(i)
        elif(methods[i] == '3'):
            df = normalization(i)
        elif(methods[i] == '4'):
            df = standardization(i)
        elif(methods[i] == '5'):
            df = robust_scaling(i)
    return render(request, 'webpages/results/resPreprocessingFeatureScaling.html')
def absolute_maximum_scale(i):
    try:
        df[col_name_num[i]]=df[col_name_num[i]]/df[col_name_num[i]].max()
    except:
        pass
    return df
def min_max_scale(i):
    try:
        if((df[col_name_num[i]].max()-df[col_name_num[i]].min())!=0):#denominator should not be 0
            df[col_name_num[i]]=(df[col_name_num[i]]-df[col_name_num[i]].min())/(df[col_name_num[i]].max()-df[col_name_num[i]].min())
    except:
        pass
    return df
def normalization(i):
    try:
        if((df[col_name_num[i]].max()-df[col_name_num[i]].min())!=0):
            df[col_name_num[i]]=(df[col_name_num[i]]-df[col_name_num[i]].mean())/(df[col_name_num[i]].max()-df[col_name_num[i]].min())
    except:
        pass    
    return df
def standardization(i):
    try:
        df[col_name_num[i]]=(df[col_name_num[i]]-df[col_name_num[i]].mean())/df[col_name_num[i]].std()
    except:
        pass
    return df
def robust_scaling(i):
    try:
        df[col_name_num[i]]=(df[col_name_num[i]]-df[col_name_num[i]].median())/(df[col_name_num[i]].quantile(0.75)-df[col_name_num[i]].quantile(0.25))
    except:
        pass
    return df



def preprocessingFeatureEncoding(request):
    return render(request, 'webpages/preprocessingFeatureEncoding.html')
def resPreprocessingFeatureEncoding(request):
    methods = (request.POST['Emet']).split()    
    n = int(request.POST['n'])
    global col_name_numerical
    col_name_numerical = []
    for j in numericaldata:
        j = int(j)
        col_name_numerical.append(col_name[j])
    try:
        col_name_numerical.remove(col_name_all[targetVariable])
    except:
        pass
    global col_name_all_enc
    col_name_all_enc = col_name_all
    try:
        col_name_all_enc.remove(col_name_all[targetVariable])
    except:
        pass
    for i in col_name_numerical:
        try:
            col_name_all_enc.remove(i)
        except:
            pass
    for i in range(len(methods)):
        i = int(i)
        if(methods[i] == '0'):
            pass
        elif(methods[i] == '1'):
            df = label_encoding(i)
        elif(methods[i] == '2'):
            df = one_hot_encoding(i)
        elif(methods[i] == '3'):
            df = hash_encoding(i, n)
        elif(methods[i] == '4'):
            df = one_hot_encoding_many_cat(i, n)
        elif(methods[i] == '5'):
            df = frequency_encoding(i)
        elif(methods[i] == '6'):
            df = mean_encoding(i)
    return render(request, 'webpages/results/resPreprocessingFeatureEncoding.html')
def label_encoding(i):
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    df[col_name_all_enc[i]+"_label_encoder"]= label_encoder.fit_transform(df[col_name_all_enc[i]]) #make new column
    df.drop([col_name_all_enc[i]], axis=1,inplace=True)   #delete the old column 
    return df
def one_hot_encoding(i):
    dummies = pd.get_dummies(df[col_name_all_enc[i]],drop_first=True)  
    for j in list(dummies.columns):
        df["ohe_"+j]=dummies[j]
    df.drop([col_name[i]], axis=1,inplace=True)   #delete the old column
    return df
def hash_encoding(i,n):
    import category_encoders as ce
    encoder=ce.HashingEncoder(cols=col_name_all_enc[i],n_components=n)
    data_encoder=encoder.fit_transform(df[col_name_all_enc[i]])
    col_name_temp=data_encoder.columns.tolist()
    for j in col_name_temp:
        df["he_"+col_name_all_enc[i]+j]=data_encoder[j]
    df.drop([col_name_all_enc[i]], axis=1,inplace=True)   #delete the old column
    return df
def one_hot_encoding_many_cat(i,n): #n means n top most repeating categories
    lst_n=df[col_name_all_enc[i]].value_counts().sort_values(ascending=False).head(n).index
    lst_n=list(lst_n)
    for categories in lst_n:
        df[categories]=np.where(df[col_name_all_enc[i]]==categories,1,0)
    df.drop([col_name_all_enc[i]], axis=1,inplace=True)   #delete the old column
    return df
def frequency_encoding(i):
    x=df[col_name_all_enc[i]].value_counts().to_dict()
    df["fe_"+col_name_all_enc[i]]=df[col_name_all_enc[i]].map(x)
    df.drop([col_name_all_enc[i]], axis=1,inplace=True)   #delete the old column
    return df
def mean_encoding(i):
    temp = "mean_orderinal_"+col_name_all_enc[i]
    mean_ordinal=df.groupby([col_name_all_enc[i]])[col_name[int(targetVariable)]].mean().to_dict()
    df[temp]=df[col_name_all_enc[i]].map(mean_ordinal)
    df.drop([col_name_all_enc[i]], axis=1,inplace=True)   #delete the old column
    return df









def preprocessingFeatureSelection(request):
    return render(request, 'webpages/preprocessingFeatureSelection.html')


def resPreprFV(request):
    return render(request, 'webpages/results/resPrepFV.html')


def resPreprFC(request):
    return render(request, 'webpages/results/resPrepFC.html')


def resPreprFMC(request):
    return render(request, 'webpages/results/resPrepFMC.html')


def resPreprFMR(request):
    return render(request, 'webpages/results/resPrepFMR.html')


def resPreprFA(request):
    return render(request, 'webpages/results/resPrepFA.html')


def resPreprER(request):
    return render(request, 'webpages/results/resPrepER.html')


def resPreprEL(request):
    return render(request, 'webpages/results/resPrepEL.html')


