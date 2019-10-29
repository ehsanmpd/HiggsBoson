""" implemetation of helper functions that are used in 5 main algorithms"""
""" Authors: Mohammad, Ehsan, Fereshte """
""" Date: 19 October 2019 """

import numpy as np
import sys
sys.path.insert(0, 'scripts/')
from proj1_helpers import *
import csv
import os
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def data_analysis_splitting(Train_dataset, Test_dataset,TRAINING_DATA, TESTING_DATA):
    """
        This function is defined to split the datasets based on the number of jets.
        Then, each number of jet is divided into two sections (depends on whether it has value for "DER_mass_MMC" or not). 
        Finally, for all eight sub-datasets, we revome features which all have -999 values. 
    """

    print('... Start the datasets analysing and splitting ...')
    print('... Loading datasets. It might take a while. ')

    # Loading train and test datasets as well as hearders:
    
    headers = headers_reader(Train_dataset)
    y_train, tx_train, ids_train = load_csv_data(Train_dataset)
    y_test, tx_test, ids_test = load_csv_data(Test_dataset)
    
    
    os.mkdir('data/sub_datasets')
    
    # Datasets (train and test sets) are splitted base on the number of jets.
    # Since the number of jests is recorded in column 22, we split datasets based on the values on that column.
    # After splitting, column 22 will be deleted.  

   
    
    for jet in range(4):
        
        print("... Splitting datasets with number of jets equal to {0:d}".format(jet))
        
        # Splitting train set and removing the column 22:
        
        tx_jet_train = tx_train[tx_train[:, 22] == jet]
        
        y_jet_train = y_train[tx_train[:, 22] == jet]
        ids_jet_train = ids_train[tx_train[:, 22] == jet]
        tx_jet_train = np.delete(tx_jet_train, 22, 1)
        
        # splitting train set and removing the column 22:
        
        tx_jet_test = tx_test[tx_test[:, 22] == jet]
        
        y_jet_test = y_test[tx_test[:, 22] == jet]
        ids_jet_test = ids_test[tx_test[:, 22] == jet]
        tx_jet_test = np.delete(tx_jet_test, 22, 1)

        # Deleting column 24 from headers (In headers, column one and two are Id and predictions, repectively):
        
        headers_jet = np.delete(headers, 24)
        
        # Finding features which contain only -999 value:
        
        features_jet = np.ones(tx_jet_train.shape[1], dtype=bool)
        header_features_jet = np.ones(tx_jet_train.shape[1] + 2, dtype=bool)
        
        for i in range(tx_jet_train.shape[1]):
            temp = tx_jet_train[:, i]
            number_nan = len(temp[temp == -999])
            if number_nan == len(temp):
                features_jet[i] = False
                header_features_jet[i + 2] = False

                
        # For the number of jet equal to zero, we make two changes: 1. removing the outliers. 2. removing the last column which is full of zero.
        
        if jet == 0:
            
            # 1. removing the outliers:
            
            to_remove = (tx_jet_train[:, 3] < 200)
            tx_jet_train = tx_jet_train[to_remove, :]
            y_jet_train = y_jet_train[to_remove]
            ids_jet_train = ids_jet_train[to_remove]

            #2. removing the last column which is full of zero:
            
            features_jet[-1] = False
            header_features_jet[-1] = False
        
        
        # Deleting columns with only -999 value
        
        tx_jet_train = tx_jet_train[:, features_jet]
        tx_jet_test = tx_jet_test[:, features_jet]
        headers_jet = headers_jet[header_features_jet]

        # Finding -999 values in the mass
        
        nan_mass_jet_train = (tx_jet_train[:, 0] == -999)
        nan_mass_jet_test = (tx_jet_test[:, 0] == -999)
       

        header_nan_mass_jet = np.ones(len(headers_jet), dtype=bool)
        header_nan_mass_jet[2] = False

        # Writing all the eight sub datasets for train and test sets: 
        
        write_data(TRAINING_DATA[2 * jet], y_jet_train[nan_mass_jet_train], tx_jet_train[nan_mass_jet_train, :][:, 1:],
                   ids_jet_train[nan_mass_jet_train], headers_jet[header_nan_mass_jet], 'train')

        write_data(TRAINING_DATA[2 * jet + 1], y_jet_train[~nan_mass_jet_train], tx_jet_train[~nan_mass_jet_train, :],
                   ids_jet_train[~nan_mass_jet_train], headers_jet, 'train')

        write_data(TESTING_DATA[2 * jet], y_jet_test[nan_mass_jet_test], tx_jet_test[nan_mass_jet_test, :][:, 1:],
                   ids_jet_test[nan_mass_jet_test], headers_jet[header_nan_mass_jet], 'test')

        write_data(TESTING_DATA[2 * jet + 1], y_jet_test[~nan_mass_jet_test], tx_jet_test[~nan_mass_jet_test, :],
                   ids_jet_test[~nan_mass_jet_test], headers_jet, 'test')

    print("... Splitting datasets is done...")

#--------------------

def headers_reader(data_path):
    """
       Loading headers
    """

    f = open(data_path, 'r')
    reader = csv.DictReader(f)
    headers = reader.fieldnames

    return headers

#--------------------

def write_data(output, y, tx, ids, headers, type_):
    """
        Writing sub data into CSV files
    """
    
    with open(os.path.join('data/sub_datasets', output), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=headers)
        writer.writeheader()
        if type_ == 'train':
            for r1, r2, r3 in zip(ids, y, tx):
                if r2 == 1:
                    pred = 's'
                elif r2 == -1:
                    pred = 'b'
                else:
                    pred = r2
                dic = {'Id': int(r1), 'Prediction': pred}
                for i in range(len(r3)):
                    dic[headers[i + 2]] = float(r3[i])
                writer.writerow(dic)
        elif type_ == 'test':
            for r1, r3 in zip(ids, tx):
                dic = {'Id': int(r1), 'Prediction': '?'}
                for i in range(len(r3)):
                    dic[headers[i + 2]] = float(r3[i])
                writer.writerow(dic)

#--------------------

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    """
    Inputs
    ----------
        y: ndarray
            1D array of the correct labels. 
        tx: ndarray
            2D array of the training data.
        w: ndarray
            1D array of the weight vector.
         
    Outputs
    -------
        grad: ndarray
            1D array of the gradient.
        error: ndarray
            1D array of the error for each training data.
    """
    error = y - tx.dot(w)
    grad = -tx.T.dot(error) / len(error)
    return grad, error

#-----------------------------------------------------------

def compute_mse(e):
    """Compute the mse for vector e."""
    """
    Input
     ----------
        e: ndarray
            1D array of the error vector.
         
    Output
    -------
        mse: float
            Mean squared error.
    """
    return 1/2*np.mean(e**2)

#------------------------------------------------------------

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generates a minibatch iterator for a dataset.

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    Inputs
    ----------
        y: ndarray
            1D array containing the correct labels of the training data.
        tx: ndarray
            2D array containing the training data.
        batch_size: int
            Integer representing the size of the batch returned by the iterator.
        num_batches: int
            Integer representing the number of batches to be returned.
        shuffle: bool
            Boolean value indicating if the data should be shuffled when returned as mini batches.

    Outputs
    ------
        tuple
             Tuple containing the next batch of training examples for gradient descent in the form (labels, training_data).
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

#---------------------------------------------------------------

def standardize(x):
    """ Standardization """
    """
    Input
    ------
        x: ndarray
        1D array of input
    
    Output
    ------
        std_data: ndarray
            1D array that represents the standardization of i nput
    """
    std_data = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return std_data

#---------------------------------------------------------------------

def create_polynomial(tX,degree):
    """Create polynomial features based on the input degree: 1 + X + X^2 + X^3 + ...+ X^degree"""
    """
    Inputs
    ---------- 
        tX: ndarray
            2D array of the training data.
        degree: integer
            An integer showing the maximum degree.

    Output
    -------
        polynomial: ndarray
            2D array of the polynomial augmented training data.
        """
    polynomial = np.ones((len(tX), 1))
    for deg in range(1, degree+1):
        polynomial = np.c_[polynomial, np.power(tX, deg)]
    return polynomial

#--------------------

def add_features_multi(tX,original_feature_size):
    """
    It makes xi*xj (i!=j, j<i)
    """
    tX_augmented=tX.copy()
    for i in range(1,original_feature_size+1):
        for j in range (i+1,original_feature_size+1):
            feature = tX.T[i] * tX.T[j]
            feature = np.reshape(feature, (1,tX.shape[0])).T
            tX_augmented = np.c_[tX_augmented, feature]
    return tX_augmented

def add_features_multi_degree2(tX,original_feature_size):
    """
    It makes (xi*xj)^2 (i!=j, j<i)
    """
    tX_augmented=tX.copy()
    for i in range(1,original_feature_size+1):
        for j in range (i+1,original_feature_size+1):
            feature = (tX.T[i] * tX.T[j]) ** 2
            feature = np.reshape(feature, (1,tX.shape[0])).T
            tX_augmented = np.c_[tX_augmented, feature]
    return tX_augmented


def add_features_multi_sqrt(tX,original_feature_size):
    """
    It makes sqrt(|xi*xj|) (i!=j, j<i)
    """
    tX_augmented=tX.copy()
    for i in range(1,original_feature_size+1):
        for j in range (i+1,original_feature_size+1):
            mul = abs(tX.T[i] * tX.T[j])
            feature = np.sqrt(mul)
            feature = np.reshape(feature, (1,tX.shape[0])).T
            tX_augmented = np.c_[tX_augmented, feature]
    return tX_augmented


def add_features_multi_sin(tX,original_feature_size):
    """
    It makes sin(xi*xj) (i!=j, j<i)
    """
    tX_augmented=tX.copy()
    for i in range(1,original_feature_size+1):
        for j in range (i+1,original_feature_size+1):
            feature = np.sin(tX.T[i]*tX.T[j])
            feature = np.reshape(feature, (1,tX.shape[0])).T
            tX_augmented = np.c_[tX_augmented, feature]
    return tX_augmented



def add_features_multi_arctan(tX,original_feature_size):
    """
    It makes arctan(xi*xj) (i!=j, j<i)
    """
    tX_augmented=tX.copy()
    for i in range(1,original_feature_size+1):
        for j in range (i+1,original_feature_size+1):
            mul = np.arctan(tX.T[i]*tX.T[j])
            feature = mul
            feature = np.reshape(feature, (1,tX.shape[0])).T
            tX_augmented = np.c_[tX_augmented, feature]
    return tX_augmented

#--------------------

# it returns the proper test and train sets based on the fold number and chunk size, for cross validation
"""a test function to get the accuracy of logistic regression; will be removed in the final version"""
def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    # y_pred = np.dot(data, weights)
    y_pred = sigmoid(data.dot(weights))
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred

#--------------------    
# Since the evaluation will be based on -1 and 1, now we will predict based on that

def predict_labels(weights, data):
    
    """Generates class predictions given weights, and a test data matrix"""
    
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred

#--------------------
    
def create_csv_submission(ids, y_pred, name):
    
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})

#--------------------

def build_k_indices(y, k_fold):
    """
        Build k-indices for the Cross-Validation
    """
    
    number_of_row = int(y.shape[0] / k_fold)
    #print("number of rows: ",y.shape[0])
    indices = np.random.permutation(y.shape[0])
    #print("indices: ",indices.shape)
    k_indices = [indices[k * number_of_row: (k + 1) * number_of_row] for k in range(k_fold)]
    return np.array(k_indices)

#--------------------

def sort_predictions (y_pred, ids_pred, TESTING_DATA):
    ids = []
    pred = []

    idx = min(ids_pred[:][0])


    length = np.sum(len(i) for i in y_pred)

    print("Concatenate the predictions.")

    # Sortting the final data based on idx:

    for i in range(length):
        for j in range(len(TESTING_DATA)):
            if len(ids_pred[j]) > 0:
                if ids_pred[j][0] == idx:
                    ids.append(idx)
                    pred.append(y_pred[j][0])
                    ids_pred[j] = np.delete(ids_pred[j], 0)
                    y_pred[j] = np.delete(y_pred[j], 0)
                    break

        if i % 100000 == 0: ### it prints the steps of concatenation.
            print(u'  {0:d}/{1:d} concatenated'.format(i, length))

        idx += 1

    return np.array(pred), np.array(ids)

#-----------------------------------------------------------------

def compute_RR(y, tx, w):
    """
        Compute the RMSE cost given the the inputs for the cost.
    """

    return np.sqrt(2 * compute_cost_RR(y, tx, w))

#--------------------

def compute_cost_RR(y, tx, w):
    """
        Compute the MSE cost.

        INPUT:
            y           - Predictions vector
            tx          - Samples
            w           - Weights

        OUTPUT:
            cost        - Double value for the costs seen in the course.
    """
    # Compute the error
    e = y - tx.dot(w)

    # Compute the cost
    return 1. / 2 * np.mean(e ** 2)

#--------------

def percentage_of_correct_prediction_logistic(y, tx, w_star):
    """
        Return the percentage of correct predictions (between -1 and 1)
    """
    pred = sigmoid(tx.dot(w_star))
    if (len(y.shape) == 1):
        y.shape = (-1, 1)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    right = np.sum(pred == y)
    wrong = len(pred) - right

    return 1-(float(wrong) / float(len(pred)))

#-----------------

def percentage_of_correct_prediction(y, tx, w_star):
    """
        Return the percentage of correct predictions (between -1 and 1)
    """

    pred = np.dot(tx, w_star)

    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    right = np.sum(pred == y)
    wrong = len(pred) - right

    return 1-(float(wrong) / float(len(pred)))

#-----------------
