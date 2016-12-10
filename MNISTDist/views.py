from django.shortcuts import render
from .models import Device,MiniBatch
from django.utils import timezone
import datetime
import numpy 
from django.http import FileResponse
from .ourfunctions import *
from django.http import HttpResponse
import os

path=os.getcwd()

def imalive(request):
    '''
    Device will POST his model of phone to this address when it's ready and alive
    The server will send back the ID assigned to this device and the URL for the dataset
    '''
    if request.method == 'POST': #POST because device is sending its type(model)
        idNum = Device.objects.count() + 1
        Device.objects.create(deviceID=idNum, deviceModel=request.body, connection_time=timezone.now(), lastActiveTime=timezone.now(), totalDataSetsGiven=0, totalDataSetsRelevant=0, avgComputingTime=0)
        dataSetURL = "None" # if we want we can put here a URL that the deivce will download the data from
        return HttpResponse("ID: " + str(idNum) + " " + "train-images: " + dataSetURL)

def getNeuralNet(request):
    '''
    return neural network model - neuralNet
    '''
    response=FileResponse(open(getPrivateNeuralNet(), 'rb'))
    response['Content-Disposition'] = 'attachment; filename=nerualNetFile.npz'
    return response

def getTrainSet(request):
    '''
    return training set. in our case, returns train.npz of MNIST
    '''
    response=FileResponse(open(path+r"\MNIST_data_set\train.npz", 'rb'))
    response['Content-Disposition'] = 'attachment; filename=train.npz'
    return response

def getTestSet(request):
    '''
    return test set. in our case, returns test.npz of MNIST
    '''
    response=FileResponse(open(path+r"\MNIST_data_set\test.npz", 'rb'))
    response['Content-Disposition'] = 'attachment; filename=test.npz'
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
            (isTrain ,subsetDataForDevice, minibatchID, epochNumber, isTestset, isFinished) = getSubsetData(devID)
            calculateStats(Device.objects.get(deviceID = devID), minibatchID, epochNumber)
            tempFilePath=path+r"\files4runtime\Data.npz"
            numpy.savez(tempFilePath, isTrain = isTrain, minibatchID = minibatchID, epochNumber = epochNumber, 
                        subsetDataForDevice = subsetDataForDevice, isTestset = isTestset, isFinished = isFinished)
            response=FileResponse(open(tempFilePath, 'rb'))
            response['Content-Disposition'] = 'attachment; filename=Data.npz'
            return response

def postData(request):
    '''
    Device will POST its deviceID, epoch number, computation time and training/validation results
    Server will not send anything back
    '''
    if request.method == 'POST': #POST because device is sending the parameters mentioned above
        (devID,miniBatchID, epochNumber, computingTime, computedResult,accuracy) = parsePostDataParameters(request.body)
        currentDevice = Device.objects.get(deviceID = devID)
        currentMiniBatch = MiniBatch.objects.get(minibatchID=miniBatchID)
        if dataIsRelevant(currentDevice,currentMiniBatch): #check if data from device is relevant (server didn't drop its result for irrelevence-if too much time has passed)
            #updating stats
            currentDevice.lastActiveTime=timezone.now()
            currentComputTime=currentMiniBatch.startComputingTime-timezone.now()
            currentDevice.avgComputingTime=(currentDevice.avgComputingTime*currentDevice.totalDataSetsRelevant+currentComputTime.seconds)/(currentDevice.totalDataSetsRelevant+1)
            currentDevice.totalDataSetsRelevant=currentDevice.totalDataSetsRelevant+1
            currentDevice.save()
            currentMiniBatch.status=2
            currentMiniBatch.accuracy=accuracy
            currentMiniBatch.deviceComputingTime=computingTime
            currentMiniBatch.finishComputingTime=timezone.now()
            currentMiniBatch.save()
            if currentMiniBatch.isTrain:
                updateNeuralNet(computedResult)
            else :
                updateEpochStats(computedResult)
            if currentMiniBatch.isFromTestset:
                if MiniBatch.Objects.filter(isFromTestset=False).exclude(status=2).count() == 0:
                    updateTestsetStats(computedResult)
        return HttpResponse("None")

def dumpDataBase(request):
    #dump devices
    with open(path+r"\files4runtime" + "\Device.txt", "w") as myfile:
        myfile.write("deviceID deviceModel totalDataSetsGiven totalDataSetsRelevant\n")
        for device in Device.objects.all():
            myfile.write(str(device.deviceID)+" "+str(device.deviceModel)+" "+str(device.totalDataSetsGiven)+" "+str(device.totalDataSetsRelevant)+"\n")
    #dump minibatches
    with open(path+r"\files4runtime" + "\MiniBatch.txt", "w") as myfile:
        myfile.write("minibatchID\tepochID\t isTrain isFromTestset deviceID status deviceComputingTime accuracy \n")
        for minibatch in MiniBatch.objects.all():
            myfile.write(str(minibatch.minibatchID)+"\t"+str(minibatch.epochID)+"\t"+str(minibatch.isTrain).lower()+" "+str(minibatch.isFromTestset).lower()+" "+str(minibatch.deviceID if minibatch.deviceID else -1)+" "+str(minibatch.status)+" "+str(minibatch.deviceComputingTime if minibatch.deviceComputingTime else -1)+" "+str(minibatch.accuracy if minibatch.accuracy else -1)+"\n")
    #dump epochs
    with open(path+r"\files4runtime" + "\Epoch.txt", "w") as myfile:
        myfile.write("epochID isTestEpoch hitRate \n")
        for epoch in Epoch.objects.all():
            myfile.write(str(epoch.epochID)+" "+str(epoch.isTestEpoch)+" "+str(epoch.hitRate)+"\n") 
    return HttpResponse("dump completed. find your files at cwd()\files4runtime")