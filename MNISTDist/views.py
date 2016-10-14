from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index(request):
    return HttpResponse("index test")
def home(request):
    return HttpResponse("home page. here we can display stats about our network and connected devices")
def imalive(request):
    return HttpResponse("here we give the device unique ID and send it to getdataset")
def getdataset(request):
    return HttpResponse("here we supply the device with link to dataset (or just id of the inputs from the dataset)")
def sendresults(request):
    return HttpResponse("here we recive results from the device")