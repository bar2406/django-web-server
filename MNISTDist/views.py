from django.shortcuts import render
from .models import Device,MiniBatch
from django.utils import timezone
import datetime
import numpy 
from django.http import FileResponse
from .ourfunctions import *
from django.http import HttpResponse

#path=r"D:\ProjectA"
path=r"C:\temp"

# Create your views here.
def home(request):
    print("here we go\n")
    if request.method == 'POST':
        print("here we go again\n")
        print(request.body)
    return HttpResponse("home page. here we can display stats about our network and connected devices")

def imalive(request):
    '''
    Device will POST his model of phone to this address when it's ready and alive
    The server will send back the ID assigned to this device and the URL for the dataset
    '''
    if request.method == 'POST': #POST because device is sending its type(model)
        idNum = Device.objects.count() + 1
        #Device.objects.create(deviceID=idNum, deviceModel=request.body, connection_time=timezone.now(), lastActiveTime=timezone.now(), numOfDataSetsGiven=0,  AvgTrainingTime=0, AvgValTime=0, minibatchID=None, epoch=None)
        Device.objects.create(deviceID=idNum, deviceModel=request.body, connection_time=timezone.now(), lastActiveTime=timezone.now(), totalDataSetsGiven=0, AvgTrainingTime=0, AvgValTime=0, minibatchID=1, epoch=1)
        dataSetURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" #TODO - give it the real URL that is needed, maybe more URLs are needed
        return HttpResponse("ID: " + str(idNum) + " " + "train-images: " + dataSetURL)

def getNeuralNet(request):
    '''
    return neural network model - neuralNet
    '''
    response=FileResponse(open(getPrivateNeuralNet(), 'rb'))
    response['Content-Disposition'] = 'attachment; filename=NeuralNet.npz'
    return response


def getData(request): 
    '''
    Device will POST his ID to this address when it had finished downloading the dataset or when after "postData" stage
    Server will send back: subset that the device will work on, epoch number, is it trainning or validation, module
    '''
    if request.method == 'POST': #POST because device is sending its ID
        try:
            devID =  int(request.body)
        except:
                raise RuntimeError("devID in getData is not int, request.body is: " + str(request.body))
        else:
            (isTrain ,subsetDataForDevice, minibatchID, epochNumber) = getSubsetData(devID)
            calculateStats(Device.objects.get(deviceID = devID), minibatchID, epochNumber)
            tempFilePath=path+r"\Data.npz"
            numpy.savez(tempFilePath, isTrain = isTrain, minibatchID = minibatchID, epochNumber = epochNumber, subsetDataForDevice = subsetDataForDevice)
            response=FileResponse(open(tempFilePath, 'rb'))
            response['Content-Disposition'] = 'attachment; filename=Data.npz'
            return response
            '''
            open on the recieving side should be as simple as: --look on the example for the correct way

            import urllib.request
            # Download the file from `url` and save it locally under `file_name`:
            urllib.request.urlretrieve(url, file_name)
            '''
            #return HttpResponse("isTrain: " + isTrain + " minibatchID: " + minibatchID + " epochNumber: " + epochNumber + " subsetDataForDevice: " + subsetDataForDevice + " neuralNet: " + neuralNet) #TODO - send subsetDataForDevice and neuralNet as files and not like this

def postData(request):
    '''
    Device will POST its deviceID, epoch number, computation time and training/validation results
    Server will not send anything back
    '''
    if request.method == 'POST': #POST because device is sending the parameters mentioned above
        (devID, epochNumber, computingTime, computedResult) = parsePostDataParameters(request.body)
        currentDevice = Device.objects.get(deviceID = devID)
        if dataIsRelevant(currentDevice): #TODO - check if data from device is relevant (server didn't drop its result for irrelevence-if too much time has passed)
            currentMiniBatch=MiniBatch.objects.get(minibatchID = currentDevice.minibatchID)
            currentMiniBatch.status=2
            if currentMiniBatch.isTrain:
                updateNeuralNet(computedResult)
            else :
                updateEpochStats(computedResult)
        currentDevice.lastActiveTime=timezone.now()
        #AvgTrainingTime=models.FloatField() 	#average minibatch training time. 
        #AvgValTime=models.FloatField() 	#average minibatch validation time. 
        return HttpResponse("thanks")
