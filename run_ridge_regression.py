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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# store output predictions:
output_file = "output/predictions.csv"
file_path = "data/sub_datasets/"

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

# Name of the training sub data:
TRAINING_DATA = ['Trainset_jet_0_without_mass.csv', 'Trainset_jet_0_with_mass.csv',
             'Trainset_jet_1_without_mass.csv', 'Trainset_jet_1_with_mass.csv',
             'Trainset_jet_2_without_mass.csv', 'Trainset_jet_2_with_mass.csv',
             'Trainset_jet_3_without_mass.csv', 'Trainset_jet_3_with_mass.csv']

# Name of the test sub data:
TESTING_DATA = ['Testset_jet_0_without_mass.csv', 'Testset_jet_0_with_mass.csv',
            'Testset_jet_1_without_mass.csv', 'Testset_jet_1_with_mass.csv',
            'Testset_jet_2_without_mass.csv', 'Testset_jet_2_with_mass.csv',
            'Testset_jet_3_without_mass.csv', 'Testset_jet_3_with_mass.csv']

# splitting data into 8 groups
data_analysis_splitting(Train_dataset, Test_dataset, TRAINING_DATA, TESTING_DATA)

## Training parameters:
k_fold = 2
lambdas = np.logspace(-15, 5, 20) # Define the lambdas values
poly_degrees = range(5,15)


y_pred = []
ids_pred = []

for model in range(len(TRAINING_DATA)):
    print("... Starting data analysis of model: %s/%s" % (model+1,len(TRAINING_DATA)))

    Y_train, X_train, Ids_train = load_csv_data(file_path+TRAINING_DATA[model])
    X_train_standard = standardize(X_train)

    lambda_star, degree_opt = compute_opt_hyper_parameters_ridge_regression(Y_train,X_train_standard, k_fold,lambdas, poly_degrees)
    print(lambda_star, degree_opt)
    print("Degree_star: % 2d  Lambda_star: % 10.3E "%(degree_opt,lambda_star))

    #### Making the polynomials based on the optimum degree and cross terms.
    X_train_standard_poly = create_polynomial(X_train_standard, degree_opt)
    X_train_standard_poly = add_cross_term_degree2(X_train_standard_poly, X_train_standard.shape[1])
    
    initial_w = np.zeros(X_train_standard_poly.shape[1])

    w_star, loss_RR = ridge_regression(Y_train, X_train_standard_poly, lambda_star)
    
    # Getting the percentage of correct predictions:    
    Acc = percentage_of_correct_prediction(Y_train, X_train_standard_poly, w_star)
        
    print(u'  The accuracy of prediction is equal to {0:f}'.format(100. * Acc))
    
    #----test----
    print("... Starting prediction on test set")

    Y_test, X_test, Ids_test = load_csv_data(file_path+TESTING_DATA[model])
    X_test_standard = standardize(X_test)


    #### Making the polynomials based on the optimum degree and cross terms.
    X_test_standard_poly = create_polynomial(X_test_standard, degree_opt)
    X_test_standard_poly = add_cross_term_degree2(X_test_standard_poly, X_test_standard.shape[1])
    
    
    test_pred = predict_labels(w_star, X_test_standard_poly)
    y_pred.append(test_pred)  
    ids_pred.append(Ids_test)

# Merging and sorting the predictions:
pred, ids = sort_predictions (y_pred, ids_pred,TESTING_DATA)

# Writing the submission file:
create_csv_submission(ids, pred, "Final_Prediction_Ridge_Regression.csv")

print(u'Ready to be submitted!')
