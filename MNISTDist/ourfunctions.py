import numpy
import chainer
import chainer.links as L
from .models import *
from django.utils import timezone
import json
import os.path

#path=r"D:\ProjectA"
#path=r"C:\temp"
path=os.getcwd()+r"\files4runtime"
try:
    os.makedirs(path)
except:
    pass
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
    #TODO - perhaps update Device to correct minibatchID. i think its unnecessary and that Device.minibatchID is redundant field

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
    if not os.path.isfile(path+'nerualNetFile.npz'):
        neuralNet=chainer.serializers.save_npz(path+r'\nerualNetFile.npz', L.Classifier(MLP(784, 10, 10)),True)
    #chainer.serializers.save_npz(path+r"\neuralNet.npz",neuralNet)
    return path+r"\nerualNetFile.npz"


def parsePostDataParameters(rquestBody):
    '''
    Parse the data that was posted from the deivce and return the parameters (in that order):
    deviceID, epochNumber, computingTime, computedResult
    '''
    jsonDec = json.decoder.JSONDecoder()
    body_unicode = rquestBody.decode('utf-8') #this is only needed in python 3. I hope it doesn't cause issues in python 2
    jsonDec = json.decoder.JSONDecoder()
    data = jsonDec.decode(body_unicode)
    deviceID = data['deviceId']
    miniBatchID = data['miniBatchID']
    epochNumber = data['epochNumber']
    computingTime = data['computingTime']
    computingTime=0 #TODO - temp
    computedResult = data['computedResult']

    return  (deviceID,miniBatchID, epochNumber, computingTime, computedResult)

def dataIsRelevant(Device,Batch):
    '''
    decides if the results of the device are relevant. can do it based on timeout from assignment or from how many minibatches were completed since this minibatch.
        special case-if results are validation
    '''
    if Batch.status==2:
        return False    #means somebody already computed this minibatch. TODO - what happens if we allocated this minibatch twice? do we accept the first or the latter?
    if Batch.status !=1:
        raise RuntimeError('error 2606: minibatch with status=0 is somehow done....')   #just a sanity check
    if Batch.isTrain==False:
        return True     #validation is always relevant
    earlierBatches=0
    for tempbatch in MiniBatch.objects.all().order_by('startComputingTime'):    #counting how many minibatches were completed since current minibatch was issued
        if tempbatch.status==2 and tempbatch.startComputingTime>Batch.startComputingTime:
            earlierBatches=earlierBatches+1
    if earlierBatches>(5*max(Batch.epochID,5)):
        return False

    return True



def updateNeuralNet(delta):
    '''
    receives compResult which is a delta of the neuralNet and updates the neuralNet
    '''
    neuralNet=numpy.load(path+r"\nerualNetFile.npz")    #TODO - perhaps we need to use  chainer.serializers.load_npz instead of np.load
    newNeuralNet=dict(neuralNet)
    for f in neuralNet.files:
        newNeuralNet[f]=neuralNet[f]+delta[f]
    neuralNet.close()
    numpy.savez(path+r"\nerualNetFile.npz",newNeuralNet)#TODO - perhaps we need to use  chainer.serializers.load_npz instead of np.load
    return True

def updateEpochStats(compResult,sizeOfValidationMiniBatch = 1000):
    '''
    receives compResult which is hit rate and number of inputs it was calculated on and updates epoch stats in the database
    '''
    curr_epoch=Epoch.objects.order_by('-epochID')[1] #if current epoch is n, we validate the n-1 epoch
    number_of_done_val_batches=MiniBatch.objects.filter(epochID=curr_epoch.epochID).filter(isTrain=False).filter(status=2).count()
    curr_epoch.hitRate=numpy.average([curr_epoch.hitRate*(number_of_done_val_batches-1),compResult]) 
    curr_epoch.save()
    if MiniBatch.objects.filter(epochID=curr_epoch.epochID).filter(isTrain=False).exclude(status=2).count() == 0: #means all validation batches are done
        curr_epoch.finishTime=timezone.now()
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
        it just means that all of the minibathces were allocated, certain percentage of them is done
    '''
    if MiniBatch.objects.filter(status=0).filter(isTrain=1).count() !=0:
        return False    #not all bathes are allocated to devices
    if (MiniBatch.objects.filter(status=2).filter(isTrain=1).count()/MiniBatch.objects.filter(isTrain=1).count()) <0.95: #TODO - 95% is arbitrary
        return False    #less the 95% of batches are done
    return True         #all batches are allocated and more then 95% of them are done

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
        if MiniBatch.objects.filter(status=1).exclude(isTrain=True).filter(epochID=Epoch.objects.count()-1).order_by('startComputingTime').count() !=0 :
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
