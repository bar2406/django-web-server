from django.db import models
from django.utils.encoding import python_2_unicode_compatible
from django.utils import timezone
import datetime

# Create your models here.
@python_2_unicode_compatible  # only if you need to support Python 2
class Device(models.Model):
    #device stats 
    deviceID=models.IntegerField(primary_key=True)    #unique device ID
    deviceModel = models.CharField(max_length=200)  #device model i.e "Sony Xperia Z3 Compact"
    connection_time = models.DateTimeField()   #time of reciveving "im alive" signal from this device
    lastActiveTime=models.DateTimeField()   #last time this device reqeusted data
    totalDataSetsGiven=models.IntegerField()
    totalDataSetsRelevant=models.IntegerField()
    avgComputingTime=models.FloatField() 	#average minibatch training time in seconds - TODO - compute it manually if needed
    #AvgValTime=models.FloatField() 	#average minibatch validation time - TODO - compute it manually if needed

    def __str__(self):
        return str(self.deviceID) + ", " + self.deviceModel

@python_2_unicode_compatible  # only if you need to support Python 2
class MiniBatch(models.Model):
    minibatchID=models.IntegerField(primary_key=True)   #some ID of the current minibatch this device is working on. not sure its actually an int
    imageIndices=models.TextField(null=True)
    epochID=models.IntegerField()
    isTrain=models.BooleanField(default=True)
    isFromTestset=models.BooleanField(default=False)
    deviceID=models.IntegerField(null=True)    #unique device ID
    status=models.IntegerField(default=0)    #0-not asssigned 1-assigned and not completed 2-done
    startComputingTime=models.DateTimeField(null=True)  #time of assigment from server
    finishComputingTime=models.DateTimeField(null=True) #time batch is recived by server
    deviceComputingTime=models.FloatField(null=True)    #computing time as measured by device
    accuracy=models.FloatField(null=True)

    def __str__(self):
        return "minibatchID: " + str(self.minibatchID)

@python_2_unicode_compatible  # only if you need to support Python 2
class Epoch(models.Model):
    epochID=models.IntegerField(primary_key=True)   #epoch number
    isTestEpoch=models.BooleanField(default=False)
    startingTime=models.DateTimeField()
    finishTime=models.DateTimeField(null=True)
    hitRate=models.FloatField(null=True)

    def __str__(self):
        return "epochID: " + str(self.epochID)
