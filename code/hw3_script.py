from __future__ import division
import os
import csv
import numpy as np
import NB

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')
# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.

#with open(os.path.join(data_dir, 'vocabulary.csv'), 'r') as f:
#    reader = csv.reader(f)
#    vocabulary = list(x[0] for x in reader)

#Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainsmall = np.genfromtxt(os.path.join(data_dir, 'XTrainsmall.csv'), delimiter=',')
yTrainsmall = np.genfromtxt(os.path.join(data_dir, 'yTrainsmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')
beta_0=7
beta_1=5
#np.shape(xTrain)
#
#XTrain = np.array([[1,1,0,0,0,1,0,0],[0,1,1,0,1,1,0,1],[0,0,1,0,0,1,1,1]])
#yTrain = np.array([1,1,0])
#XTest = np.array([[1,1,0,0,1,1,0,0],[0,1,1,0,0,1,1,0]])
#yTest = np.array([1,0,1])
#beta_0=7
#beta_1=5

# TODO: Test logProd function, defined in NB.py

x=[np.log(np.e**3),np.log(np.e**4)]

print(NB.logProd(x))

# TODO: Test NB_XGivenY function, defined in NB.py
print('XGivenY:')
D = NB.NB_XGivenY(XTrain,yTrain,beta_0,beta_1)
print(D)

# TODO: Test NB_YPrior function, defined in NB.py

print('yPrior:')
p=NB.NB_YPrior(yTrain)
print(p)

# TODO: Test NB_Classify function, defined in NB.py

yHat=NB.NB_Classify(D,p,XTest)
print('Predicted y')
print(yHat)

# TODO: Test classificationError function, defined in NB.py

print('Error')
error = NB.classificationError(yHat,yTest)
print(beta_0)
print(beta_1)
print(error)
# TODO: Run experiments outlined in HW2 PDF
