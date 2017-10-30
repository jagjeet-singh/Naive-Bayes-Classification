from __future__ import division
import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	## Outputs ##
	# log_product - float

	log_product = np.sum(x)
	return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):

	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
	## Outputs ##
	# D - (2 by V) numpy ndarray
    V = XTrain.shape[1]
    
    
    D = np.ones((2,V))
    temp = np.ones(XTrain.shape)
    temp = XTrain*yTrain[:,np.newaxis]
    
    non_zero_word_count=np.array((np.ones((1,temp.shape[1]))))
    
    for i in range(0,temp.shape[1]):
        non_zero_word_count[0,i]=np.array(np.count_nonzero(temp[:,i]))

    count_x_1_y_1 = np.array(non_zero_word_count)
    
    count_x_1_y_0 = np.array(np.ones((1,XTrain.shape[1])))
    for i in range(0,XTrain.shape[1]):
        count_x_1_y_0[0,i]=np.count_nonzero(XTrain[:,i])-non_zero_word_count[0,i]
    
    count_x_0_y_1 = np.array(np.count_nonzero(yTrain)-non_zero_word_count)    
    
    count_x_0_y_0 = np.array(XTrain.shape[0]-count_x_1_y_1-count_x_1_y_0-count_x_0_y_1)               

    D[0,:]=(count_x_1_y_0+beta_0-1)/((count_x_1_y_0+beta_0-1)+(count_x_0_y_0+beta_1-1))
    D[1,:]=(count_x_1_y_1+beta_0-1)/((count_x_1_y_1+beta_0-1)+(count_x_0_y_1+beta_1-1))     
    return D
#updated version 9/16

# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float
    num_positive_reviews = np.count_nonzero(yTrain)
    num_negative_reviews = yTrain.shape[0]-np.count_nonzero(yTrain)
    p = (num_negative_reviews/(num_positive_reviews+num_negative_reviews))

    return p
# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m
    yHat = np.ones(XTest.shape[0])
    D_log = np.log(D)
    one_minus_D_log = np.log(1-D)
    x_1_y_0 = np.sum(XTest*D_log[0,:],1)
    x_1_y_1 = np.sum(XTest*D_log[1,:],1)
    x_0_y_0 = np.sum((1-XTest)*one_minus_D_log[0,:],1)
    x_0_y_1 = np.sum((1-XTest)*one_minus_D_log[1,:],1)
    
    y_0_log = np.log(p)
    y_1_log = np.log(1-p)
    
    pred_ratios_0_by_1 = y_0_log-y_1_log+x_1_y_0-x_1_y_1+x_0_y_0-x_0_y_1
    
    yHat = 1*(pred_ratios_0_by_1<0.5)

    return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float

    error = np.sum(yHat!=yTruth)/yHat.shape[0]

    return error
