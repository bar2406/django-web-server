import numpy
import chainer
import chainer.links as L
from .models import Device


#defining neuralNet, should be static and global(?) variable
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l4=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

neuralNet = L.Classifier(MLP(784, 10, 10))


"""
getSubsetData() - init minibatch database and adds another epoch to database if: this is the first epoch or if validation is done
                    in init we need to randomly order the dataset into (training+validation) minibatches and add them to the minibatch database. all are init with MiniBatch.status=0
                    return next available minibatch or, if 95% of minibatches are done and all of the minibatches are assigned, return validation minibatch. update MiniBatch.status to 1
getNeuralNet() - return parameters of the neural network. TODO - in what format????
dataIsRelevant(Device) - decides if the results of the device are relevant. can do it based on timeout from assignment or from how many minibatches were completed since this minibatch.
                            special case-if results are validation
updateNeuralNet(compResult) - revices compResult which is a delta of the neuralNet and updates the neuralNet
    updateEpochStats(compResult) - revices compResult which is hit rate and number of inputs it was calculated on and updates epoch stats in the database
"""
#(isTrain ,subsetDataForDevice, minibatchID, epochNumber) = getSubsetData()
def getSubsetData():
    isTrain=1
    subsetDataForDevice=[1, 3, 5, 7]
    minibatchID=1
    epochNumber=1
    return isTrain ,subsetDataForDevice, minibatchID, epochNumber

def getNeuralNet():
    return neuralNet

def dataIsRelevant(Device):
    return True

def updateNeuralNet(compResult):
    #DOES SOMTHING
    return True

def updateEpochStats(compResult):
    #does somthing
    return True