import numpy
import chainer
import chainer.links as L
from .models import *
from django.utils import timezone
import json
import os.path
import chainer

path=os.getcwd()+r"\files4runtime"
try:
    os.makedirs(path)
except:
    pass

MNIST_DATASET_SIZE=60000    #for robustness, it is possiable to import the data base and checking its size
MNIST_TESTSET_SIZE=10000
TOTAL_NUMBER_OF_TRAINING_EPOCHS=10

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

def getSubsetData(DeviceID):
    '''
    init minibatch database and adds another epoch to database if: this is the first epoch or if validation is done
    in init we need to randomly order the dataset into (training+validation) minibatches and add them to the minibatch database. all are init with MiniBatch.status=0
    return next available minibatch or, if 95% of minibatches are done and all of the minibatches are assigned, return validation minibatch. update MiniBatch.status to 1

    return parameters as:
    (isTrain ,subsetDataForDevice, minibatchID, epochNumber) = getSubsetData()
    '''
    if Epoch.objects.count() == 0 or checkEpochDone() == True:
        isTestset=False
        if (Epoch.objects.count() == TOTAL_NUMBER_OF_TRAINING_EPOCHS):
            isTestset=True #finished training, start testing
        if (Epoch.objects.count() <= TOTAL_NUMBER_OF_TRAINING_EPOCHS):
            _initEpoch(isTestset)
            _initMiniBatches(isTestset)

    currentBatch=_fetchNextMiniBatch()
    if currentBatch is None:
        #No more bathcs, finished running
        return False,0,0,0,False,True #All the return parameters are not important, except the last one, which is finished running

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
    isFromTestset=currentBatch.isFromTestset
    return isTrain, subsetDataForDevice, minibatchID, epochNumber, isFromTestset, False #last parameters means - didn't finish to run

def getPrivateNeuralNet():
    ##changable constans:
    MIDDLE_LAYER_SIZE = 100
    
    ####################################################################
	
	
    if not os.path.isfile(path+r'\neuralNetFile.npz'):
        neuralNet=chainer.serializers.save_npz(path+r'\neuralNetFile.npz', L.Classifier(MLP(784, MIDDLE_LAYER_SIZE, 10)),True)
    return path+r"\neuralNetFile.npz"


def parsePostDataParameters(rquestBody):
    '''
    Parse the data that was posted from the deivce and return the parameters (in that order):
    deviceID, epochNumber, computingTime, computedResult
    '''
    body_unicode = rquestBody.decode('utf-8') #this is only needed in python 3
    jsonDec = json.decoder.JSONDecoder()
    data = jsonDec.decode(body_unicode)
    deviceID = data['deviceId']
    miniBatchID = data['miniBatchID']
    epochNumber = data['epochNumber']
    computingTime = data['computingTime']
    computingTime=data['computingTime']
    computedResult = data['computedResult']
    accuracy = data['accuracy']


    return  (deviceID,miniBatchID, epochNumber, computingTime, computedResult,accuracy)

def dataIsRelevant(Device,Batch):
    '''
    decides if the results of the device are relevant. can do it based on timeout from assignment or from how many minibatches were completed since this minibatch.
        special case-if results are validation
    '''
    if Batch.status==2:
        return False    #means somebody already computed this minibatch
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
    ##changable constans:
    MIDDLE_LAYER_SIZE = 100
    
    ####################################################################
    neuralNet=L.Classifier(MLP(784, MIDDLE_LAYER_SIZE, 10))
    chainer.serializers.load_npz(path+r"\neuralNetFile.npz",neuralNet)
    neuralNet.predictor.l1.W.data=neuralNet.predictor.l1.W.data+numpy.array(delta['predictor/l1/W']).astype(numpy.float32)
    neuralNet.predictor.l1.b.data=neuralNet.predictor.l1.b.data+numpy.array(delta['predictor/l1/b']).astype(numpy.float32)
    neuralNet.predictor.l2.W.data=neuralNet.predictor.l2.W.data+numpy.array(delta['predictor/l2/W']).astype(numpy.float32)
    neuralNet.predictor.l2.b.data=neuralNet.predictor.l2.b.data+numpy.array(delta['predictor/l2/b']).astype(numpy.float32)
    neuralNet.predictor.l3.W.data=neuralNet.predictor.l3.W.data+numpy.array(delta['predictor/l3/W']).astype(numpy.float32)
    neuralNet.predictor.l3.b.data=neuralNet.predictor.l3.b.data+numpy.array(delta['predictor/l3/b']).astype(numpy.float32)
    chainer.serializers.save_npz(path+r"\neuralNetFile.npz",neuralNet)

    return True

def updateEpochStats(compResult,epoch):
    '''
    receives compResult which is hit rate and number of inputs it was calculated on and updates epoch stats in the database
    '''
    
    curr_epoch=Epoch.objects.get(epochID=epoch)
    #print(" current hitrate: "+str(curr_epoch.hitRate))
    number_of_done_val_batches=MiniBatch.objects.filter(epochID=curr_epoch.epochID).filter(isTrain=False).filter(isFromTestset=False).filter(status=2).count()
    curr_epoch.hitRate=(curr_epoch.hitRate*(number_of_done_val_batches-1)+compResult)/number_of_done_val_batches
    if MiniBatch.objects.filter(epochID=curr_epoch.epochID).filter(isTrain=False).filter(isFromTestset=False).exclude(status=2).count() == 0: #means all validation batches are done
        curr_epoch.finishTime=timezone.now()
    curr_epoch.save()
    #print("epoch: "+str(curr_epoch.epochID)+" accuracy:"+str(compResult)+" number of done val batch: "+str(number_of_done_val_batches)+" current hitrate: "+str(curr_epoch.hitRate))
    return True

def updateTestsetStats(computedResult):
    Epoch.objects.filter(isTestEpoch=False)[0].finishTime=timezone.now()


def calculateStats(deviceObj, minibatchID, epochNumber):
    '''
    calculates different statistics
    '''
    deviceObj.lastActiveTime = timezone.now()
    deviceObj.totalDataSetsGiven = deviceObj.totalDataSetsGiven + 1
    deviceObj.minibatchID = minibatchID
    deviceObj.epoch = epochNumber

def _initEpoch(isTestset):
    '''
    creat and initialize the next epoch with standart stats
    '''
    Epoch.objects.create(epochID = Epoch.objects.count()+1,isTestEpoch = isTestset, startingTime = timezone.now(), finishTime = None, hitRate = 0) #TODO - finishTime,hitRate are unknown. better if they would be None

def _creatMiniBatch(imageIndices,epochID,isTrain,isTestset):
    '''
    create and initialize MiniBatch 
    '''
    MiniBatch.objects.create(minibatchID = MiniBatch.objects.count()+1, imageIndices = json.dumps(imageIndices.tolist()), epochID = epochID, isTrain = isTrain,isFromTestset = isTestset, deviceID = None, status = 0, startComputingTime = None)

def _initMiniBatches(isTestset,batchsize = 1000,valSize = 5000): #valSize = validate set size
    '''
    creates all minibatches for latest epoch.
    randomly orders mnist into minibatches, and devides them into training and validation batches.
    '''
    if isTestset:
        testIndices = numpy.arange(MNIST_TESTSET_SIZE)
        for i in range(int(numpy.ceil(MNIST_TESTSET_SIZE/batchsize))):
            _creatMiniBatch(testIndices[:batchsize],Epoch.objects.count(),False,isTestset)
            testIndices=testIndices[batchsize:]
    else:
        order=numpy.random.permutation(MNIST_DATASET_SIZE)
        for i in range(int(numpy.ceil((MNIST_DATASET_SIZE-valSize)/batchsize))): #create Training batches
            _creatMiniBatch(order[:batchsize],Epoch.objects.count(),True,isTestset)
            order=order[batchsize:]
        for i in range(int(numpy.ceil((valSize)/batchsize))): #create Validation batches
            _creatMiniBatch(order[:batchsize],Epoch.objects.count(),False,isTestset)
            order=order[batchsize:]

def checkEpochDone():
    '''
    return true if latest epoch is done. false otherwise. validation starts when epoch is done
    p.s even if epoch is done it dosen't mean that we dont allow new deltas from that epoch.
        it just means that all of the minibathces were allocated, certain percentage of them is done
    '''
    ##changable constans:
    FINISHED_BATCHES_PRECENT = 0.95 # 95% is arbitrary, can be changed
    
    ####################################################################
    if MiniBatch.objects.filter(status=0).filter(isTrain=1).count() !=0:
        return False    #not all batches are allocated to devices
    if (MiniBatch.objects.filter(status=2).filter(isTrain=1).count()/MiniBatch.objects.filter(isTrain=1).count()) < FINISHED_BATCHES_PRECENT:
        return False    #less then FINISHED_BATCHES_PRECENT of batches are done
    return True         #all batches are allocated and more then FINISHED_BATCHES_PRECENT of them are done

def _fetchNextMiniBatch():
    '''
    returns next minibatch for processing. returns only from latest epoch
    p.s as long as checkEpochDone() dosen't decide to move to the next epoch, this function will continue to distribute batches.
        so its checkEpochDone() responsibility to make sure that we dont serve the same batch to 20 devices only becuase its the last batch in the epoch
    '''
    #1 priority: unassigned validation batches from previous epoch
    if Epoch.objects.count() > 1:
        if MiniBatch.objects.filter(status=0).exclude(isTrain=True).exclude(isFromTestset=True).exclude(epochID=Epoch.objects.count()).count() !=0 :
            return MiniBatch.objects.filter(status=0).exclude(isTrain=True).exclude(isFromTestset=True).exclude(epochID=Epoch.objects.count()).order_by('minibatchID')[0]

    #2 priority: assigned (and not done) validation batches from previous epoch that are over X time old
    if Epoch.objects.count() > 1:
        if MiniBatch.objects.filter(status=1).exclude(isTrain=True).exclude(isFromTestset=True).exclude(epochID=Epoch.objects.count()).order_by('startComputingTime').count() !=0 :
            tempBatch=MiniBatch.objects.filter(status=1).exclude(isTrain=True).exclude(isFromTestset=True).exclude(epochID=Epoch.objects.count()).order_by('startComputingTime')[0]
            delta=timezone.now()-tempBatch.startComputingTime
            if(delta.seconds>60*15): #means X=15 minutes
                return tempBatch

    #3 priority: unassigned training batches from current epoch
    if MiniBatch.objects.filter(status=0).filter(isTrain=True).filter(epochID=Epoch.objects.count()).count() != 0:
        return MiniBatch.objects.filter(status=0).filter(isTrain=True).filter(epochID=Epoch.objects.count()).order_by('minibatchID')[0]

    #4 priority: assigned (and not done) training batches from current epoch
    if MiniBatch.objects.filter(status=1).filter(isTrain=True).filter(epochID=Epoch.objects.count()).count() != 0:
        return MiniBatch.objects.filter(status=1).filter(isTrain=True).filter(epochID=Epoch.objects.count()).order_by('minibatchID')[0]

    #5 priority: finish test set epoch (unassigned test minibatches)
    if MiniBatch.objects.filter(status=0).filter(isTrain=False).filter(epochID=Epoch.objects.count()).count() != 0:
        return MiniBatch.objects.filter(status=0).filter(isTrain=False).filter(epochID=Epoch.objects.count()).order_by('minibatchID')[0]

    #6 priority: finish test set epoch (assigned and not done test minibatches)
    if MiniBatch.objects.filter(status=1).filter(isTrain=False).filter(epochID=Epoch.objects.count()).count() != 0:
        return MiniBatch.objects.filter(status=1).filter(isTrain=False).filter(epochID=Epoch.objects.count()).order_by('minibatchID')[0]

    return None
