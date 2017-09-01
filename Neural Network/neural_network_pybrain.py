import pandas as pd
from time import time
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError

bbc = pd.read_csv('supervisedlearningdataset_13082016.csv', parse_dates=True)

# converting NaN values in the dataset to 0.
for column in bbc.columns:
    bbc[column] = bbc[column].fillna(0)

# removing document id from dataset.
bbc = bbc.drop("Document id", 1)

# table information.
no_of_input_features = bbc.shape[1]-1
no_of_inputs = bbc.shape[0]

# colomn names.
feature_columns = bbc.columns[0:no_of_input_features]

# target variable.
target = "label"

# variables of neural network.
no_of_hidden_layers = 1
no_of_outputs = 1

# creating a neural network. It is basic sigmoid function neural network
net = buildNetwork(no_of_input_features, no_of_hidden_layers, no_of_outputs)

# creating a dataset for neural network.
ds = SupervisedDataSet(no_of_input_features, no_of_outputs)

# adding elements to neural network dataset.
for i in range(no_of_inputs):
    ds.addSample(bbc[feature_columns][i:i+1], bbc[target][0])

# splitting data into training and test sets.
testData, trainData = ds.splitWithProportion(.2)

# creating a trainer with neural network and dataset.
trainer = BackpropTrainer(net, trainData)
# training the BackpropTrainer on the dataset.
t0 = time()
print "training error:" + str(trainer.train())        # training for one epoch.
t1 = time() - t0
print "time taken: " + str(t1)

trnResult = percentError(trainer.testOnClassData(),
                         trainData[target])
tstResult = percentError(trainer.testOnClassData(
                    dataset = testData), testData[target])

print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
