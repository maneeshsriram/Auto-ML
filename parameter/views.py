from django.shortcuts import render
from django.core.files.storage import default_storage


def tuningDataset(request):
    return render(request, 'webpages/parameterDataset.html')


def tuningPkl(request):
    if request.method == 'POST':
        global targetvariable
        global Dataset_file_name
        global Pkl_file_name
        targetvariable = request.POST['tar']
        file1 = request.FILES['csvfile']
        file2 = request.FILES['pklfile']
        Dataset_file_name = default_storage.save(file1.name, file1)
        Pkl_file_name = default_storage.save(file2.name, file2)
    print(targetvariable, Pkl_file_name, Dataset_file_name)
    return render(request, 'webpages/parameterPkl.html')


def tuningRegression(request):
    return render(request, 'webpages/parameterRegression.html')

def tuningClassification(request):
    return render(request, 'webpages/parameterClassification.html')

def tuningMethod(request):
    global model
    global metric 
    model = request.POST['mod']
    metric = request.POST['met']
    return render(request, 'webpages/parameterMethod.html')


