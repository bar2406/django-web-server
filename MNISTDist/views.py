from django.shortcuts import render
from .models import Device
from django.utils import timezone
import datetime

# Create your views here.
from django.http import HttpResponse

def index(request):
    return HttpResponse("index test")
def home(request):
    print("here we go\n")
    if request.method == 'POST':
        print("here we go again\n")
        print(request.body)
    return HttpResponse("home page. here we can display stats about our network and connected devices")
def imalive(request):
    if request.method == 'POST': #POST because device is sending its type
        idNum = Device.objects.count() + 1
        #Device.objects.create(deviceID=idNum, deviceModel=request.body, connection_time=timezone.now(), lastActiveTime=timezone.now(), numOfDataSetsGiven=0,  AvgTrainingTime=0, AvgValTime=0, minibatchID=None, epoch=None)
        Device.objects.create(deviceID=idNum, deviceModel=request.body, connection_time=timezone.now(), lastActiveTime=timezone.now(), totalDataSetsGiven=0, AvgTrainingTime=0, AvgValTime=0, minibatchID=1, epoch=1)
        dataSetURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" #TODO - give it the real URL that is needed, maybe more URLs are needed
        return HttpResponse("ID: " + str(idNum) + " " + "train-images: " + dataSetURL)
    
def getdataset(request):
    return HttpResponse("here we supply the device with link to dataset (or just id of the inputs from the dataset)")
def sendresults(request):
    return HttpResponse("here we recive results from the device")