""" implemetation of all steps to run the project in sequential"""
""" Authors: Mohammad, Ehsan, Fereshte """
""" Date: 20 October 2019 """

import numpy as np
import sys

sys.path.insert(0, 'scripts/')
sys.path.insert(0, 'data/')
from helper_functions import *
from implementations import *
from proj1_helpers import *

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Check the train and test datasets exist in the right place (data folder).
Train_dataset = 'data/train.csv'
try:
    open(Train_dataset, 'r')
except:
    raise NameError('Cannot open the train data set %s! Please make sure it exists in the data folder')

Test_dataset = 'data/test.csv'
try:
    open(Test_dataset, 'r')
except:
    raise NameError('Cannot open the test data set %s! Please make sure it exists in the data folder')


max_iters_training = 400
gamma_training = 0.1
max_iters_cross_validation = 100
gamma_cross_validation = 0.1

y_pred = []
ids_pred = []

np.random.seed(1)

# Splitting all the data into two parts: training (0.8) and testing (0.2):
y, tX, ids = load_csv_data(Train_dataset)
X_train_standard = standardize(tX)

num_train_data = 0.8 * len(y)

N = len(y)
N_train =int(N*0.8)
N_test = N - N_train
D = len(X_train_standard[0])

ids_train = ids[0:N_train]
ids_test = ids[N_train:N]

tX_train = X_train_standard[0:N_train]
tX_test = X_train_standard[N_train:N]

y_train = y[0:N_train]
y_test = y[N_train:N]


'''test of gradient decent'''
gamma_GD = 0.1
max_iters_GD = 300
test_GD(tX_train,y_train,tX_test,y_test,gamma_GD,max_iters_GD)


'''test of stochastic gradient decent'''
gamma_SGD = 0.001
max_iters_SGD = 500
test_SGD(tX_train,y_train,tX_test,y_test,gamma_GD,max_iters_GD)

'''test of least square'''
test_LS(tX_train,y_train,tX_test,y_test)


'''test of ridge regression'''
lambda_RR=1e-5
test_ridge_regression(tX_train,y_train,tX_test,y_test,lambda_RR)

'''test of logistic regression'''
gamma_LR = 0.5
max_iters_LR = 500
test_logistic_regression(tX_train,y_train,tX_test,y_test,gamma_LR,max_iters_LR)

'''test of logistic ridge regression'''
gamma_LR = 0.5
max_iters_LR = 500
lambda_LRR=1e-5
test_logistic_ridge_regression(tX_train,y_train,tX_test,y_test,gamma_LR,max_iters_LR,lambda_LRR)

