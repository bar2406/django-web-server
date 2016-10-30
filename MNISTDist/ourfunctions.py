import numpy
import chainer
import chainer.links as L
from .models import *
from django.utils import timezone
import json
import os.path

#path=r"D:\ProjectA"
path=r"C:\temp"
MNIST_DATASET_SIZE=60000    #TODO - for robustness, perhaps actually importing the data base and checking its size 
#defining neuralNet, should be static and global(?) variable
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

'''class neuralNet:
    neuralNetArg = None
    def __init__(self):
        self.neuralNetArg = L.Classifier(MLP(784, 10, 10))
'''



def getSubsetData(DeviceID):
    '''
    init minibatch database and adds another epoch to database if: this is the first epoch or if validation is done
    in init we need to randomly order the dataset into (training+validation) minibatches and add them to the minibatch database. all are init with MiniBatch.status=0
    return next available minibatch or, if 95% of minibatches are done and all of the minibatches are assigned, return validation minibatch. update MiniBatch.status to 1

    return parameters as:
    (isTrain ,subsetDataForDevice, minibatchID, epochNumber) = getSubsetData()
    '''
    if Epoch.objects.count() == 0 or checkEpochDone() == True:
        _initEpoch()
        _initMiniBatches()
    currentBatch=_fetchNextMiniBatch()

    #updating currentBatch stats
    currentBatch.deviceID=DeviceID
    currentBatch.status=1
    currentBatch.startComputingTime=timezone.now()
    currentBatch.save()

    #create return valuse
    isTrain=currentBatch.isTrain
    jsonDec = json.decoder.JSONDecoder()
    subsetDataForDevice=jsonDec.decode(currentBatch.imageIndices)
    minibatchID=currentBatch.minibatchID
    epochNumber=currentBatch.epochID
    return isTrain ,subsetDataForDevice, minibatchID, epochNumber

def getPrivateNeuralNet():
	#TODO - this will recreate the net every time. this is bad, neuralNet should be static. no idea how to do that. cant have it as a class though.
	#it needs to be of type L.Classifier(MLP(784, 10, 10). noam this is YOUR PROBLEM :-)
    if not os.path.isfile('nerualNetFile.npz'):
        neuralNet=chainer.serializers.save_npz(path+r'\nerualNetFile.npz', L.Classifier(MLP(784, 10, 10)),True)
    #chainer.serializers.save_npz(path+r"\neuralNet.npz",neuralNet)
    return path+r"\nerualNetFile.npz"


def parsePostDataParameters(rquestBody):
    '''
    Parse the data that was posted from the deivce and return the parameters (in that order):
    deviceID, epochNumber, computingTime, computedResult
    '''
    jsonDec = json.decoder.JSONDecoder()
    data=jsonDec.decode(rquestBody)
    deviceID = data['deviceId']
    epochNumber = data['epochNumber']
    computingTime = data['computingTime']
    #tempFilePath=path+r"\Data.npz"
    computedResult = data[computedResult'']

    #update NeuralNet
    
    return (deviceID, epochNumber, computingTime, computedResult)

def dataIsRelevant(Device):
    '''
    decides if the results of the device are relevant. can do it based on timeout from assignment or from how many minibatches were completed since this minibatch.
        special case-if results are validation
    '''
    return True

def updateNeuralNet(compResult):
    '''
    receives compResult which is a delta of the neuralNet and updates the neuralNet
    '''
    #DOES SOMTHING
    return True

def updateEpochStats(compResult):
    '''
    receives compResult which is hit rate and number of inputs it was calculated on and updates epoch stats in the database
    '''
    #does somthing
    return True

def calculateStats(deviceObj, minibatchID, epochNumber):
    '''
    calculates different statistics
    '''
    deviceObj.lastActiveTime = timezone.now()
    deviceObj.totalDataSetsGiven = deviceObj.totalDataSetsGiven + 1
    deviceObj.minibatchID = minibatchID
    deviceObj.epoch = epochNumber

def _initEpoch():
    '''
    creat and initialize the next epoch with standart stats
    '''
    Epoch.objects.create(epochID = Epoch.objects.count()+1, startingTime = timezone.now(), finishTime = timezone.now(), hitRate = 0) #TODO - finishTime,hitRate are unknown. better if they would be None

def _creatMiniBatch(imageIndices,epochID,isTrain):
    '''
    create and initialize MiniBatch 
    '''
    MiniBatch.objects.create(minibatchID = MiniBatch.objects.count()+1, imageIndices = json.dumps(imageIndices.tolist()), epochID = epochID, isTrain = isTrain, deviceID = None, status = 0, startComputingTime = None) 

def _initMiniBatches(batchsize = 1000,valSize = 5000):
    '''
    creates all minibatches for latest epoch.
    randomly orders mnist into minibatches, and devides them into training and validation batches.
    '''
    order=numpy.random.permutation(MNIST_DATASET_SIZE)
    for i in range(int(numpy.ceil((MNIST_DATASET_SIZE-valSize)/batchsize))): #create Training batches
        _creatMiniBatch(order[:batchsize],Epoch.objects.count(),True)
        order=order[batchsize:]
    for i in range(int(numpy.ceil((valSize)/batchsize))): #create Validation batches
        _creatMiniBatch(order[:batchsize],Epoch.objects.count(),False)
        order=order[batchsize:]

def checkEpochDone():
    '''
    return true if latest epoch is done. false otherwise. validation starts when epoch is done
    p.s even if epoch is done it dosen't mean that we dont allow new deltas from that epoch.
        it just means that all of the minibathces were allocated, certien percentage of them is done 
        (and all validation results are done?)
    '''
    return False

def _fetchNextMiniBatch():
    '''
    returns next minibatch for processing. returns only from latest epoch
    p.s as long as checkEpochDone() dosen't decide to move to the next epoch, this function will continue to distribute batches.
        so its checkEpochDone() responsibility to make sure that we dont serve the same batch to 20 devices only becuase its the last batch in the epoch
    '''
    #1 priority: unassigned validation batches from previous epoch
    if Epoch.objects.count() > 1:
        if MiniBatch.objects.filter(status=0).exclude(isTrain=True).filter(epochID=Epoch.objects.count()-1).count() !=0 :
            return MiniBatch.objects.filter(status=0).exclude(isTrain=True).filter(epochID=Epoch.objects.count()-1).order_by('minibatchID')[0]

    #2 priority: assigned (and not done) validation batches from previous epoch that are over X time old
    if Epoch.objects.count() > 1:
        tempBatch=MiniBatch.objects.filter(status=1).exclude(isTrain=True).filter(epochID=Epoch.objects.count()-1).order_by('startComputingTime')[0]
        delta=timezone.now()-tempBatch.startComputingTime
        if(delta.seconds>60*15): #means X=15
            return tempBatch

    #3 priority: unassigned training batches from current epoch
    if MiniBatch.objects.filter(status=0).filter(isTrain=True).filter(epochID=Epoch.objects.count()).count() != 0:
        return MiniBatch.objects.filter(status=0).filter(isTrain=True).filter(epochID=Epoch.objects.count()).order_by('minibatchID')[0]

    #4 priority: assigned (and not done) training batches from current epoch
    if MiniBatch.objects.filter(status=1).filter(isTrain=True).filter(epochID=Epoch.objects.count()).count() != 0:
        return MiniBatch.objects.filter(status=1).filter(isTrain=True).filter(epochID=Epoch.objects.count()).order_by('minibatchID')[0]

    raise RuntimeError('error 1234: didn\'t found MiniBatch')
