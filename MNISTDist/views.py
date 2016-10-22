from django.shortcuts import render
from .models import Device
from django.utils import timezone
import datetime
import numpy 
from django.http import FileResponse
# from .ourfunctions import updateEpochStats,updateNeuralNet,dataIsRelevant,getNeuralNet,getSubsetData
from ourfunctions import *
from django.http import HttpResponse


# Create your views here.

def index(request):
    return HttpResponse("index test")

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
        print("hi " + str(idNum) + " bye")
        #Device.objects.create(deviceID=idNum, deviceModel=request.body, connection_time=timezone.now(), lastActiveTime=timezone.now(), numOfDataSetsGiven=0,  AvgTrainingTime=0, AvgValTime=0, minibatchID=None, epoch=None)
        Device.objects.create(deviceID=idNum, deviceModel=request.body, connection_time=timezone.now(), lastActiveTime=timezone.now(), totalDataSetsGiven=0, AvgTrainingTime=0, AvgValTime=0, minibatchID=1, epoch=1)
        dataSetURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" #TODO - give it the real URL that is needed, maybe more URLs are needed
        return HttpResponse("ID: " + str(idNum) + " " + "train-images: " + dataSetURL)

def getData(request): 
    '''
        Device will POST his ID to this address when it had finished downloading the dataset or when after "postData" stage
        Server will send back: subset that the device will work on, epoch number, is it trainning or validation, module
    '''
    if request.method == 'POST': #POST because device is sending its ID
        try:
            devID =  int(request.body)
        except:
            print("devID in getData is not int, request.body is: " + request.body)
        else:
            """
            getSubsetData() - init minibatch database and adds another epoch to database if: this is the first epoch or if validation is done
                                in init we need to randomly order the dataset into (training+validation) minibatches and add them to the minibatch database. all are init with MiniBatch.status=0
                              return next available minibatch or, if 95% of minibatches are done and all of the minibatches are assigned, return validation minibatch. update MiniBatch.status to 1
            getNeuralNet() - return parameters of the neural network. TODO - in what format???? = NPZ format
            """
            print devID
            (isTrain ,subsetDataForDevice, minibatchID, epochNumber) = getSubsetData()
            neuralNet = getNeuralNet()  
            #TODO - insert it to a function that does statistics
            calculateStats(Device.objects.get(deviceID = devID), minibatchID, epochNumber)
            #currentDevice = Device.objects.get(deviceID = devID)
            #currentDevice.lastActiveTime = timezone.now()
            #currentDevice.totalDataSetsGiven = currentDevice.totalDataSetsGiven + 1
            #currentDevice.minibatchID = minibatchID
            #currentDevice.epoch = epochNumber
            tempFilePath="D:\ProjectA\Data.npz"
            numpy.savez(tempFilePath, isTrain = isTrain, minibatchID = minibatchID, epochNumber = epochNumber, subsetDataForDevice = subsetDataForDevice, neuralNet = neuralNet)
            response=FileResponse(open(tempFilePath, 'rb'))
            response['Content-Disposition'] = 'attachment; filename=Data.npz'
            return response
            '''
            open on the recieving side should be as simple as:

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

    """
    dataIsRelevant(Device) - decides if the results of the device are relevant. can do it based on timeout from assignment or from how many minibatches were completed since this minibatch.
                                special case-if results are validation
    updateNeuralNet(compResult) - revices compResult which is a delta of the neuralNet and updates the neuralNet
     updateEpochStats(compResult) - revices compResult which is hit rate and number of inputs it was calculated on and updates epoch stats in the database
    """
    if request.method == 'POST': #POST because device is sending the parameters mentioned above
        (deviceID, epochNumber, compTime, compResult) = parsePostDataParameters(request.body)
        currentDevice = Device.objects.get(deviceID = devID)
        if dataIsRelevant(currentDevice): #TODO - check if data from device is relevant (server didn't drop its result for irrelevence-if too much time has passed)
            currentMiniBatch=MiniBatch.objects.get(minibatchID = currentDevice.minibatchID)
            currentMiniBatch.status=2
            if currentMiniBatch.isTrain:
                updateNeuralNet(compResult)
            else :
                updateEpochStats(compResult)
        currentDevice.lastActiveTime=timezone.now()
        #AvgTrainingTime=models.FloatField() 	#average minibatch training time. 
        #AvgValTime=models.FloatField() 	#average minibatch validation time. 
        return HttpResponse("thanks")
