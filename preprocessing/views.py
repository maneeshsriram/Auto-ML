from django.shortcuts import render
import pandas as pd
import numpy as np
import random
from django.http import HttpResponse
import csv
from io import BytesIO
import base64



def preprocessingForm(request):
    return render(request, 'webpages/preprocessing.html')
def preprocessingCol(request):
    if request.method == 'POST':
        global numericaldata
        global col_names_cat
        global col_names_num
        global col_names_all
        global col_name
        global df
        col_names_num = []
        numericaldata = (request.POST['num']).split()
        file = request.FILES['csvfile'] 
        df = pd.read_csv(file)
        col_name = df.columns.tolist()
        col_names_all = df.columns.tolist()
        for i in numericaldata:
            col_names_num.append(col_name[int(i)])
        for col in col_names_num:
            for j in range(df.shape[0]):
                s=df[col][j]
                s_new=''
                try:
                    for i in s:
                        if(i=='1' or i=='2' or i=='3' or i=='4' or i=='5'or i=='6' or i=='7' or i=='8' or i=='9' or i=='0' or i=='.'):
                            s_new=s_new + i
                    s_new=float(s_new)
                    df[col][j]=s_new 
                except:
                    pass
        col_names_cat = col_name.copy()
        for i in col_names_num:
            try:
                col_names_cat.remove(i)
            except:
                pass
    return render(request, 'webpages/preprocessingCol.html')



#missing values
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
        df.dropna(inplace=True)
    except:
        pass
    df.reset_index(inplace=True, drop=True)
    return df
def drop_col_explicit(col_name, i):
    try:
        df.drop([col_name[i]], axis=1, inplace=True)
    except:
        pass
    try:
        numericaldata.remove(str(i))
    except:
        pass
    try:
        col_names_num.remove(col_name[i])
    except:
        pass
    try:
        col_names_cat.remove(col_name[i])
    except:
        pass
    try:
        col_name.remove(col_name[i])
    except:
        pass
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




#Outlier
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
        # col_name_outl = []
        # for j in numericaldata: 
        #     j = int(j)
        #     col_name_outl.append(col_name[j])
        q1 = np.percentile(df[col_names_num[i]], 25)
        q3 = np.percentile(df[col_names_num[i]], 75)
        IQR = q3-q1
        lwr_bound = q1-(1.5*IQR)
        upr_bound = q3+(1.5*IQR)
        size = df.shape[0]
        for j in range(size): 
            try:
                if (df[col_names_num[i]][j]<lwr_bound or df[col_names_num[i]][j]>upr_bound):
                    df.drop(j,inplace=True)
            except:
                pass
        df.reset_index(inplace = True, drop = True)
        return df
def zscore(i):
    # col_name_outl = []
    # for j in numericaldata:
    #     j = int(j)
    #     col_name_outl.append(col_name[j])
    thres = 3.3
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
    return df
def iso_forest(i):
    # col_name_outl = []
    # for j in numericaldata:
    #     j = int(j)
    #     col_name_outl.append(col_name[j])
    from sklearn.ensemble import IsolationForest
    model=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.02),random_state=2)
    model.fit(df[[col_names_num[i]]])
    df['anomaly_score'] = model.predict(df[[col_names_num[i]]])
    for j in range(0,df.shape[0]):
        try:
            if(df['anomaly_score'][j]==-1):
                df.drop(j,inplace=True)
        except:
            pass
    df.reset_index(inplace=True, drop=True)
    df.drop(['anomaly_score'],inplace=True, axis = 1)
    return df



#Feature Scaling
def preprocessingFeatureScaling(request):
    return render(request, 'webpages/preprocessingFeatureScaling.html')
def resPreprocessingFeatureScaling(request):
    methods = (request.POST['Fmet']).split()
    global targetVariable
    targetVariable = int(request.POST['tar'])
    try:
        col_names_num.remove(col_names_all[targetVariable])
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
        df[col_names_num[i]]=df[col_names_num[i]]/df[col_names_num[i]].max()
    except:
        pass
    return df
def min_max_scale(i):
    try:
        if((df[col_names_num[i]].max()-df[col_names_num[i]].min())!=0):#denominator should not be 0
            df[col_names_num[i]]=(df[col_names_num[i]]-df[col_names_num[i]].min())/(df[col_names_num[i]].max()-df[col_names_num[i]].min())
    except:
        pass
    return df
def normalization(i):
    try:
        if((df[col_names_num[i]].max()-df[col_names_num[i]].min())!=0):
            df[col_names_num[i]]=(df[col_names_num[i]]-df[col_names_num[i]].mean())/(df[col_names_num[i]].max()-df[col_names_num[i]].min())
    except:
        pass    
    return df
def standardization(i):
    try:
        df[col_names_num[i]]=(df[col_names_num[i]]-df[col_names_num[i]].mean())/df[col_names_num[i]].std()
    except:
        pass
    return df
def robust_scaling(i):
    try:
        df[col_names_num[i]]=(df[col_names_num[i]]-df[col_names_num[i]].median())/(df[col_names_num[i]].quantile(0.75)-df[col_names_num[i]].quantile(0.25))
    except:
        pass
    return df



#Feature Encoding
def preprocessingFeatureEncoding(request):
    return render(request, 'webpages/preprocessingFeatureEncoding.html')
def resPreprocessingFeatureEncoding(request):
    methods = (request.POST['Emet']).split()    
    n = int(request.POST['n'])
    global col_names_cat_copy
    try:
        col_names_cat.remove(col_names_all[int(targetVariable)])
    except:
        pass
    col_names_cat_copy = col_names_cat.copy()
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
    df[col_names_cat_copy[i]+"_label_encoder"]= label_encoder.fit_transform(df[col_names_cat_copy[i]]) #make new column
    df.drop([col_names_cat_copy[i]], axis=1,inplace=True) 
    try:
        col_names_cat.remove(col_name[i])
    except:
        pass
    try:
        col_name.remove(col_name[i])
    except:
        pass
    return df
def one_hot_encoding(i):
    dummies = pd.get_dummies(df[col_names_cat_copy[i]],drop_first=True)  
    for j in list(dummies.columns):
        df["ohe_"+j]=dummies[j]
    df.drop([col_name[i]], axis=1,inplace=True)   #delete the old column
    try:
        col_names_cat.remove(col_name[i])
    except:
        pass
    try:
        col_name.remove(col_name[i])
    except:
        pass
    return df
def hash_encoding(i,n):
    import category_encoders as ce
    encoder=ce.HashingEncoder(cols=col_names_cat_copy[i],n_components=n)
    data_encoder=encoder.fit_transform(df[col_names_cat_copy[i]])
    col_name_temp=data_encoder.columns.tolist()
    for j in col_name_temp:
        df["he_"+col_names_cat_copy[i]+j]=data_encoder[j]
    df.drop([col_names_cat_copy[i]], axis=1,inplace=True)   #delete the old column
    try:
        col_names_cat.remove(col_name[i])
    except:
        pass
    try:
        col_name.remove(col_name[i])
    except:
        pass
    return df
def one_hot_encoding_many_cat(i,n): #n means n top most repeating categories
    lst_n=df[col_names_cat_copy[i]].value_counts().sort_values(ascending=False).head(n).index
    lst_n=list(lst_n)
    for categories in lst_n:
        df[categories]=np.where(df[col_names_cat_copy[i]]==categories,1,0)
    df.drop([col_names_cat_copy[i]], axis=1,inplace=True)   #delete the old column
    try:
        col_names_cat.remove(col_name[i])
    except:
        pass
    try:
        col_name.remove(col_name[i])
    except:
        pass
    return df
def frequency_encoding(i):
    x=df[col_names_cat_copy[i]].value_counts().to_dict()
    df["fe_"+col_names_cat_copy[i]]=df[col_names_cat_copy[i]].map(x)
    df.drop([col_names_cat_copy[i]], axis=1,inplace=True)   #delete the old column
    try:
        col_names_cat.remove(col_name[i])
    except:
        pass
    try:
        col_name.remove(col_name[i])
    except:
        pass
    return df
def mean_encoding(i):
    temp = "mean_orderinal_"+col_names_cat_copy[i]
    mean_ordinal=df.groupby([col_names_cat_copy[i]])[col_name[int(targetVariable)]].mean().to_dict()
    df[temp]=df[col_names_cat_copy[i]].map(mean_ordinal)
    df.drop([col_names_cat_copy[i]], axis=1,inplace=True)   #delete the old column
    try:
        col_names_cat.remove(col_name[i])
    except:
        pass
    try:
        col_name.remove(col_name[i])
    except:
        pass
    return df








#Feature Selection
def preprocessingFeatureSelecNum(request):
    return render(request, 'webpages/preprocessingFeatureSelecNum.html')
def preprocessingFeatureSelection(request):
    global num_cols_upd
    global targetVariableUpdated
    num_cols_ = (request.POST['num_cols_upd']).split()
    num_cols_upd = []
    for i in num_cols_:
        num_cols_upd.append(df.columns.tolist()[int(i)])
    targetVariableUpdated = request.POST['tar']
    return render(request, 'webpages/preprocessingFeatureSelection.html')
def resPreprFV(request):
    dat = {}
    for i in df.var().items():
        dat[i] = i
    data = {
        "d": dat
    }
    return render(request, 'webpages/results/resPrepFV.html', data)
def resPreprFC(request):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 6))
    col_name = df.columns.tolist()
    col_names_num.append(col_name[int(targetVariableUpdated)])
    new_df = df[col_names_num].copy()
    new_df[col_name[int(targetVariableUpdated)]] = df[col_name[int(targetVariableUpdated)]]
    x = new_df.drop(labels=[col_name[int(targetVariableUpdated)]], axis=1)
    y = new_df[col_name[int(targetVariableUpdated)]]
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    X_train[col_name[int(targetVariableUpdated)]] = new_df[col_name[int(targetVariableUpdated)]]
    cor = X_train.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_png = buffer.getvalue()
    pair = base64.b64encode(img_png)
    pair = pair.decode('utf-8')
    buffer.close()
    data = {
        'plot': pair
    }
    return render(request, 'webpages/results/resPrepFC.html', data)
def resPreprFMC(request):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import mutual_info_classif
    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 6))
    col_name = df.columns.tolist()
    col_names_num.append(col_name[int(targetVariableUpdated)])
    new_df = df[col_names_num].copy()
    new_df[col_name[int(targetVariableUpdated)]] = df[col_name[int(targetVariableUpdated)]]
    x = new_df.drop(labels=[col_name[int(targetVariableUpdated)]], axis=1)
    y = new_df[col_name[int(targetVariableUpdated)]]
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    mutual_info = mutual_info_classif(X_train, y_train)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_png = buffer.getvalue()
    pair = base64.b64encode(img_png)
    pair = pair.decode('utf-8')
    buffer.close()
    data = {
        'plot': pair,
    }
    return render(request, 'webpages/results/resPrepFMC.html', data)
def resPreprFMR(request):
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.model_selection import train_test_split
    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 6))
    col_name = df.columns.tolist()
    col_names_num.append(col_name[int(targetVariableUpdated)])
    new_df = df[col_names_num].copy()
    new_df[col_name[int(targetVariableUpdated)]] = df[col_name[int(targetVariableUpdated)]]
    x = new_df.drop(labels=[col_name[int(targetVariableUpdated)]], axis=1)
    y = new_df[col_name[int(targetVariableUpdated)]]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    mutual_info = mutual_info_regression(X_train.fillna(0), y_train)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    mutual_info.sort_values(ascending=False).plot.bar(figsize=(15,5))
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_png = buffer.getvalue()
    pair = base64.b64encode(img_png)
    pair = pair.decode('utf-8')
    buffer.close()
    data = {
        'plot': pair,
    }
    return render(request, 'webpages/results/resPrepFMR.html', data)
def resPreprFA(request):
    dat = {}    
    for i in list(col_names_num):
        mean_abs_diff = np.sum(np.abs(df[i] -np.mean(df[i], axis =0 )), axis = 0)/df[i].shape[0]
        dat[(i, mean_abs_diff)] = (i)
    data = {
        'd': dat
    }
    return render(request, 'webpages/results/resPrepFA.html', data)
def resPreprER(request):
    col_name = df.columns.tolist()
    col_names_num.append(col_name[int(targetVariableUpdated)])
    new_df = df[col_names_num].copy()
    new_df[col_name[int(targetVariableUpdated)]] = df[col_name[int(targetVariableUpdated)]]
    x=new_df.drop(labels=[col_name[int(targetVariableUpdated)]],axis=1)
    y=new_df[col_name[int(targetVariableUpdated)]]
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sel.fit(X_train, y_train)
    selected_feat= X_train.columns[(sel.get_support())]
    data = {
        'total_features' : X_train.shape[1],
        'selected_features_len' : len(selected_feat),
        'selected_features': list(selected_feat)
    }
    return render(request, 'webpages/results/resPrepER.html', data)
def resPreprEL(request):
    col_name = df.columns.tolist()
    col_names_num.append(col_name[int(targetVariableUpdated)])
    new_df = df[col_names_num].copy()
    new_df[col_name[int(targetVariableUpdated)]] = df[col_name[int(targetVariableUpdated)]]
    x = new_df.drop(labels=[col_name[int(targetVariableUpdated)]], axis=1)
    y = new_df[col_name[int(targetVariableUpdated)]]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
    sel.fit(X_train, y_train)
    selected_feat = X_train.columns[(sel.get_support())]
    data = {
        'total_features': X_train.shape[1],
        'selected_features_len': len(selected_feat),
        'selected_features': list(selected_feat)
    }
    return render(request, 'webpages/results/resPrepEL.html', data)

def delCol(request):
    col_name_delete = []
    col_del = (request.POST['del_col']).split()
    for i in col_del:
        col_name_delete.append(num_cols_upd[int(i)])
    df.drop(col_name_delete, inplace=True, axis=1)
    print(df.columns.tolist())
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Preprocessed.csv"'
    writer = csv.writer(response)
    col_name_upd = df.columns.tolist()
    writer.writerow(col_name_upd)
    rows = df.to_numpy().tolist()
    for i in range(len(rows)):
        writer.writerow(rows[i])
    df.dropna(inplace=True)
    df.reset_index(inplace = True, drop = True)
    return response
