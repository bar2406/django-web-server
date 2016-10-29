"""import datetime
from django.db import models
from django.utils.encoding import python_2_unicode_compatible
from django.utils import timezone
# Create your models here.

@python_2_unicode_compatible  # only if you need to support Python 2
class device(models.Model):

    #device stats
    #deviceID=models.IntegerField()    #unique device ID
    #devicemodel = models.CharField(max_length=200)  #device model i.e "Sony Xperia Z3 Compact"
    #connection_time = models.DateTimeField()   #time of reciveving "im alive" signal from this device
    #lastactivetime=models.DateTimeField()   #last time this device reqeusted data.
    #totaldatasetsgiven=models.IntegerField()    #some way of marking how much work this device has done
    
    #last minibatch data
    #minibatchID=models.IntegerField()   #some ID of the current minibatch this device is working on. not sure its actually an int
    #epoch=models.IntegerField() #from what epoch this minibatch is from
"""
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
    lastActiveTime=models.DateTimeField()   #last time this device reqeusted data.
    #numOfDataSetsGiven=models.IntegerField()    #some way of marking how much work this device has done
    totalDataSetsGiven=models.IntegerField()
    AvgTrainingTime=models.FloatField() 	#average minibatch training time. 
    AvgValTime=models.FloatField() 	#average minibatch validation time. 

    #last minibatch data
    minibatchID=models.IntegerField()   #some ID of the current minibatch this device is working on. not sure its actually an int
    #minibatchIDSQL=models.ForeignKey(MiniBatch, on_delete=models.CASCADE)   #some ID of the current minibatch this device is working on. not sure its actually an int
    epoch=models.IntegerField() #from what epoch this minibatch is from

    def __str__(self):
        return str(self.deviceID) + ", " + self.deviceModel

@python_2_unicode_compatible  # only if you need to support Python 2
class MiniBatch(models.Model):
    minibatchID=models.IntegerField(primary_key=True)   #some ID of the current minibatch this device is working on. not sure its actually an int
    imageIndices=models.TextField(null=True)	#TODO - this can be decode using JSON. see at http://stackoverflow.com/questions/1110153/what-is-the-most-efficent-way-to-store-a-list-in-the-django-models
    epochID=models.IntegerField()
    #epochIDSQL=models.ForeignKey(Epoch, on_delete=models.CASCADE)
    isTrain=models.BooleanField(default=True)
    deviceID=models.IntegerField(null=True)    #unique device ID
    #deviceIDSQL=models.ForeignKey(Device, on_delete=models.CASCADE)
    status=models.IntegerField()    #0-not asssigned 1-assigned and not completed 2-done
    startComputingTime=models.DateTimeField(null=True)
    
    def __str__(self):
        return "minibatchID: " + str(self.minibatchID)

@python_2_unicode_compatible  # only if you need to support Python 2
class Epoch(models.Model):
    epochID=models.IntegerField(primary_key=True)   #epoch number
    startingTime=models.DateTimeField()
    finishTime=models.DateTimeField(null=True)
    hitRate=models.FloatField(null=True)

    def __str__(self):
        return "epochID: " + str(self.epochID)


    
