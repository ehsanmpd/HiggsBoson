""" Implemetation of 5 required algorithms """
""" Authors: Mohammad, Ehsan, Fereshte """
""" Date: 19 October 2019 """

import sys
sys.path.insert(0, 'scripts/')
from helper_functions import *
import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Implementation of linear regression using gradient descent """
    """
    Inputs
    ---------
        y: ndarray
            1D array of the correct labels 
        tx: ndarray
            2D array of the training data.
        initial_w: ndarray
            1D array of the initial weight vector.
        max_iters: int
            Maximum number of iterations to run the gradient descent.
        gamma: float
            Learning rate of the gradient descent.
         
    Outputs
     -------
        w: ndarray
            1D array of the final weight vector.
        loss: float
            Loss value corresponding to the final weight vector.
    """
    # Define variables to store ws and losses
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient, error, and loss
        grad, err = compute_gradient(y, tx, w)
        loss = compute_mse(err)
        # updating w
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    # return last w in column vector and loss
    return ws.pop(), losses.pop()
#------------------------------------------------------------

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Implementation of linear regression using stochastic gradient descent """
    """
    Inputs
    ----------
        y: ndarray
            1D array of the correct labels 
        tx: ndarray
            2D array of the training data.
        initial_w: ndarray
            1D array of the initial weight vector.
        max_iters: int
            Maximum number of iterations to run the gradient descent.
        gamma: float
            Learning rate of the gradient descent.
         
    Outputs
    -------
        w: ndarray
            1D array of the final weight vector.
        loss: float
            Loss value corresponding to the final weight vector.
    """
    # Define variables to store ws and losses
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # compute a stochastic gradient, error, and loss
            grad, err = compute_gradient(y_batch, tx_batch, w)
            loss = compute_mse(err)
            # updating w
            w = w - gamma * grad
            # store w and loss
            ws.append(w)
            losses.append(loss)

    # return last w in column vecvtor and loss
    return ws.pop(), losses.pop()
#---------------------------------------------------------------

def least_squares(y, tx):
    """ Implementation of least squares regression using normal equations"""
    """ 
    Inputs
    ----------
        y: ndarray
            1D array of the correct labels 
        tx: ndarray
            2D array of the training data.
            
    Outputs
    -------
        w: ndarray
            1D array of the final weight vector.
        loss: float
            Loss value corresponding to the final weight vector.
    """
    # compute optimal weights
    txT = np.transpose(tx)
    l = np.dot(txT, tx)
    r = np.dot(txT, y)
    w = np.linalg.solve(l, r)
    # calculate loss
    e = y - np.dot(tx, w)
    loss = compute_mse(e)
    
    # return w in column vector and the corresponding loss
    return w, loss
#--------------------------------------------------------------

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Implementation of logistic regression using gradient descent or SGD """
    """
    Inputs
    ----------
        y: ndarray
            1D array of the correct labels 
        tx: ndarray
            2D array of the training data.
        initial_w: ndarray
            1D array of the initial weight vector.
        max_iters: int
            Maximum number of iterations to run the gradient descent.
        gamma: float
            Learning rate of the gradient descent.
         
    Outputs
    -------
        w: ndarray
            1D array of the final weight vector.
        loss: float
            Loss value corresponding to the final weight vector.
    """
    # transform row vector to column vector
    if (len(y.shape) == 1): 
        y.shape = (-1, 1)
    if (len(initial_w.shape) == 1): 
        initial_w.shape = (-1, 1)
    # Define variables to store w, current loss, and previous loss
    w = initial_w
    curr_loss = -1
    pre_loss = -1
    # Define threshold for early stopping of gradient descent
    threshold = 1e-8
    for iter_ in range(max_iters):
        # compute loss and update w
        w, curr_loss = gradient_descent_for_logistic_reg(y, tx, w, gamma)
        # if the last two losses defference is below than the threshold, stop the gradient descent 
        if pre_loss != -1 and np.abs(curr_loss-pre_loss) < threshold:
            break
        prev_loss = curr_loss
    #return w in column vector and the corresponding loss
    return w,curr_loss
# --------------------------

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Implementation of regularized logistic regression using gradient descent or SGD """
    """
    Inputs
    ----------
        y: ndarray
            1D array of the correct labels. 
        tx: ndarray
            2D array of the training data.
        lambda_: float
            Regularization parameter.
        initial_w: ndarray
            1D array of the initial weight vector.
        max_iters: int
            Maximum number of iterations to run the gradient descent.
        gamma: float
            Learning rate.
         
    Outputs
    -------
        w: ndarray
            1D array of the last weight vector.
        loss: float
            Loss corresponding to the last weight vector.
    """
    # transform row vector to column vector
    if (len(y.shape) == 1): 
        y.shape = (-1, 1)
    if (len(initial_w.shape) == 1): 
        initial_w.shape = (-1, 1)
    # Define variables to store w, current loss, and previous loss
    w = initial_w
    curr_loss = -1
    prev_loss = -1
    # Define threshold for early stopping of gradient descent
    threshold = 1e-8
    
    for iter_ in range(max_iters):
        # update w and calculate loss
        w, curr_loss= gradient_descent_for_logistic_reg(y, tx, w, gamma, lambda_)
        # if the last two losses defference is below than the threshold, stop the gradient descent 
        if prev_loss != -1 and np.abs(curr_loss-prev_loss) < threshold:
            break
        prev_loss = curr_loss
    
    # return w in column vector form and the corresponding loss
    return w, curr_loss


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def gradient_descent_for_logistic_reg(y, tx, w, gamma,lambda_=None):
    """ Implementation of one step of gradient descent for logistic regression & computation of the loss """
    """
    Inputs
    ----------
        y: ndarray
            1D array of the correct labels 
        tx: ndarray
            2D array of the training data.
        w: ndarray
            1D array containing the weight vector.
        gamma: float
            Learning rate of the gradient descent.
    
    Outputs
    -------
        w: ndarray
            1D array of the updated weight vector.
        loss: float
            Loss corresponding to the updated weight vector.
        """
    gradient = compute_gradient_logestic_reg(y, tx, w,lambda_)
    w = w - gamma*gradient
    loss = negative_log_likelihood(y, tx, w,lambda_)
    return w, loss
#-------------------------------------------------------------

def compute_gradient_logestic_reg(y, tx, w, lambda_=None):
    """ Computes the gradient for 1) logistic regression: if lambda_ is none
                                  2) regularized logistic regression: if lambda_ is not none. """
    """
    Inputs
    ----------
        y: ndarray
            1D array of the correct labels. 
        tx: ndarray
            2D array of the training data.
        w: ndarray
            1D array of the weight vector.
        lambda_: float, default is none
            Regularization parameter for regularized logistic regression. 
        
    Outputs
    -------
        gradient: ndarray
            1D array of the gradient
    """
    # compute the basic gradient
    z = np.dot(tx, w)
    predicted = sigmoid(z)
    gradient = np.dot(np.transpose(tx), (predicted - y))
    # add the regularization term
    if lambda_ is not None:
        gradient = gradient + 2*lambda_*w
    # version fereshte: return gradient
    # changed by ehsan: return gradient/y.shape[0]
    return gradient/y.shape[0]
#-------------------------------------------------------------

def sigmoid(z):
    """ Implements sigmoid function """
    """   
    Inputs
    ----------
        z: ndarray
            Input of the sigmoid function.
         
    Outputs
    -------
        result: ndarray
            output of the sigmoid function.
    """
    result = 1.0 / (1 + np.exp(-z))
    return result
#--------------------------------------------------------------

def negative_log_likelihood(y, tx, w, lambda_=None):
    """ Compute the mean Negative Log-Likelihood loss for 1) logistic regression: if lambda_ is none
                                                          2) regularized logistic regresion: if lambda_ is not none """
    """
    Inputs
    ----------
        y: ndarray
            1D array of the correct labels. 
        tx: ndarray
            2D array of the training data.
        w: ndarray
            1D array of the weight vector.
        lambda_: float, defualt is none
            Regularization parameter for regularized logistic regression. 
         
    Output
    -------
        loss: float
            Mean negative log-likelihood loss.
        """
    # transform row vector to column vector
    if (len(y.shape) == 1): 
        y.shape = (-1, 1)
    pred = tx.dot(w)
    # compute log[1 + e^(pred)]
    temp1 = np.logaddexp(0, pred)
    # multiplication of element by element
    temp2 = np.multiply(y, pred)
    loss = np.sum(temp1-temp2)
    # add regularization term
    if lambda_ is not None:
        loss = loss + lambda_ * np.squeeze(np.dot(w.T, w))

    return loss/y.shape[0]

#---------------------------------------------------------------

def ridge_regression(y, tx, lambda_):
    """
        Use the Ridge Regression method to find the best weights

        INPUT:
            y           - Predictions
            tx          - Samples

        OUTPUT:
            w           - Best weights
            loss        - Minimum loss
    """

    # Compute optimal weights
    xx = np.dot(np.transpose(tx), tx)
    # Add the lambda on the diagonal
    bxx = xx + lambda_ * np.identity(len(xx))
    xy = np.dot(np.transpose(tx), y)
    w_star = np.linalg.solve(bxx, xy)

    loss = compute_RR(y, tx, w_star)

    return w_star, loss

#--------------------

def compute_opt_hyper_parameters(y_train,tx_train, k_fold, lambdas, poly_degrees, max_iters_cross_validation, gamma_cross_validation):
    """
    call all the training datasets and do the cross validation for each of them -------> 
    Return the best lambda for each data sets.
    
    """
    print("Cross-validation...")

    opt_lambda, opt_poly_degrees = cross_validation(y_train, tx_train, k_fold, lambdas, poly_degrees, max_iters_cross_validation, gamma_cross_validation)   

        
    return opt_lambda, opt_poly_degrees 

#------------------------------------------------------

def cross_validation(y, tx, k_fold, lambdas, poly_degrees, max_iters_cross_validation, gamma_cross_validation):

    print("  Start the %i-fold Cross Validation!..." % k_fold)

    # Spliting datasets into the k-fold subsets
    # It shuffels them as well:
    
    ## Making the shuffel and splitted indices
    
    k_indices = build_k_indices(y, k_fold)
    
    least_loss_degree = np.inf
    
    if (len(y.shape) == 1): 
        y.shape = (-1, 1)
    for power in poly_degrees:
        tx_poly = create_polynomial(tx, power)
        least_loss_lambda = np.inf
        for lamb in lambdas:
            loss_folds = 0
            for i in range(k_fold):

                y_subtest = y[k_indices[i]]
                tx_subtest  = tx_poly[k_indices[i]]

                y_subtrain  = np.delete(y,k_indices[i],axis = 0)
                tx_subtrain = np.delete(tx_poly,k_indices[i],axis = 0)

                initial_w = np.zeros(tx_subtrain.shape[1])

                if (len(y_subtrain.shape) == 1): 
                    y_subtrain.shape = (-1, 1)
                if (len(y_subtest.shape) == 1): 
                    y_subtest.shape = (-1, 1)
                if (len(initial_w.shape) == 1): 
                    initial_w.shape = (-1, 1)

                ## Finding the best w for each sub train data and lambda value

                w, loss_train =reg_logistic_regression(y_subtrain, tx_subtrain, lamb, initial_w, max_iters_cross_validation, gamma_cross_validation)

                ## Finding the loss base on the w obtained at the previous step for each sub test data and lambda value

                loss_validation = negative_log_likelihood(y_subtest, tx_subtest, w, lamb)

                loss_folds += np.abs(loss_validation) 

            if abs(loss_folds/k_fold) < abs(least_loss_lambda):

                least_loss_lambda = abs(loss_folds/k_fold)
                lambda_s = lamb
        if least_loss_lambda < least_loss_degree:
            least_loss_degree = least_loss_lambda
            degree_opt = power
            lambda_star = lambda_s
    return lambda_star,degree_opt

#---------------------

def compute_opt_hyper_parameters_ridge_regression(y_train, tx_train, k_fold, lambdas, poly_degrees):
    """
    call all the training datasets and do the cross validation for each of them ------->
    Return the best lambda for each data sets.

    """
    print("Cross-validation for ridge regression...")

    opt_lambda, opt_poly_degrees = cross_validation_ridge_regression(y_train, tx_train, k_fold, lambdas, poly_degrees,)

    return opt_lambda, opt_poly_degrees

#---------------------

def cross_validation_ridge_regression(y, tx, k_fold, lambdas, poly_degrees):

    print("  Start the %i-fold Cross Validation!..." % k_fold)

    # Spliting datasets into the k-fold subsets
    # It shuffels them as well:
    
    ## Making the shuffel and splitted indices
    
    k_indices = build_k_indices(y, k_fold)
    lambda_star=0
    degree_opt = 0
    acc_degree = 0
    if (len(y.shape) == 1): 
        y.shape = (-1, 1)
    for power in poly_degrees:
        tx_poly = create_polynomial(tx, power)
        lambda_s=0
        acc_lambda = 0
        for lamb in lambdas:
            acc = 0
            train_loss_agg = 0
            for i in range(k_fold):

                #### Here we make the the sub data sets. They have been splitted and shuffeled. 

                y_subtest = y[k_indices[i]]
                #print("y size: ",y.shape)
                #print("sub y test size: ",y_subtest.shape)
                tx_subtest  = tx_poly[k_indices[i]]
                #print("tx sub test size: ",tx_subtest.shape)

                y_subtrain  = np.delete(y,k_indices[i],axis = 0)
                tx_subtrain = np.delete(tx_poly,k_indices[i],axis = 0)
                #print("sub y  train size: ",y_subtrain.shape)
                #print("sub tx trian size: ",tx_subtrain.shape)


                #### Reshape tx and y into the correct form:

                #y_subtrain = y_subtrain.reshape((-1,))
                #tx_subtrain = tx_subtrain.reshape((-1,tx.shape[1]))

                initial_w = np.zeros(tx_subtrain.shape[1])

                if (len(y_subtrain.shape) == 1): 
                    y_subtrain.shape = (-1, 1)
                if (len(y_subtest.shape) == 1): 
                    y_subtest.shape = (-1, 1)
                if (len(initial_w.shape) == 1): 
                    initial_w.shape = (-1, 1)

                ## Finding the best w for each sub train data and lambda value

                w, loss_train = ridge_regression(y_subtrain, tx_subtrain, lamb)
                train_loss_agg += abs(loss_train)
                ## Finding the accuracy base on the w obtained at the previous step for each sub test data and lambda value
                acc += percentage_of_correct_prediction(y_subtest, tx_subtest, w)

            cur_acc = acc/k_fold
            cur_train_loss = train_loss_agg/k_fold
            if cur_acc > acc_lambda:
                lambda_s = lamb
                acc_lambda = cur_acc
            # print(
            #     'Acc_opt: ' + str(acc_degree) +'    Lambda_star: '+str(lambda_star) + '   |    cur_deg: ' + str(power) +
            #     '    cur_lambda_opt: '+ str(lambda_s) + '   cur_lambda: ' +str(lamb)+ '     cur_acc: ' + str(cur_acc) +
            #     '    cur_train_loss: ' + str(cur_train_loss))
            # print(
            #     "Acc_opt: % 2.8f  Degree_star: % 2d  Lambda_star: % 10.3E   |    acc_lambda: % 2.8f    cur_deg: % 2d    cur_lambda_opt: % 10.3E   cur_lambda: % 10.3E     cur_acc: % 2.8f" % (
            #     100 * acc_degree, degree_opt, lambda_star, 100*acc_lambda, power, lambda_s, lamb, 100 * cur_acc))
        if acc_lambda > acc_degree:
            degree_opt = power
            lambda_star = lambda_s
            acc_degree = acc_lambda
        print(
                "Acc_opt: % 2.8f  Degree_star: % 2d  Lambda_star: % 10.3E   |    acc_lambda: % 2.8f    cur_deg: % 2d    cur_lambda_opt: % 10.3E   cur_lambda: % 10.3E     cur_acc: % 2.8f" % (
                100 * acc_degree, degree_opt, lambda_star, 100*acc_lambda, power, lambda_s, lamb, 100 * cur_acc))
    return lambda_star,degree_opt

#--------------------

def test_GD(x_train, y_train, x_test, y_test,gamma,max_iters):
    """
    It runs Gradient descent method. 
    """

    print("Testing Gradient Decent...")
    initial_w = np.zeros(len(x_train[0]))
    w, loss = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
    acc = percentage_of_correct_prediction(y_test, x_test, w)
    print("Accuracy of Gradient Decent is %2.2f out of %5d train size and %5d test size"%(acc*100,len(y_train),len(y_test)))

#-----------

def test_SGD(x_train, y_train, x_test, y_test,gamma,max_iters):
    """
    It runs SGD method.
    """

    print("Testing Stochastic Gradient Decent...")

    initial_w = np.zeros(len(x_train[0]))
    w, loss = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)
    acc = percentage_of_correct_prediction(y_test, x_test, w)
    print("Accuracy of Stochastic Gradient Decent is %2.2f out of %5d train size and %5d test size"%(acc*100,len(y_train),len(y_test)))

#----------------

def test_LS(x_train, y_train, x_test, y_test):
    """
    It runs least square method.
    """

    print("Testing Least Square...")
    w, loss = least_squares(y_train, x_train)
    acc = percentage_of_correct_prediction(y_test, x_test, w)
    print("Accuracy of Least Square is %2.2f out of %5d train size and %5d test size"%(acc*100,len(y_train),len(y_test)))

#----------------

def test_ridge_regression(x_train, y_train, x_test, y_test,lambda_):
    """
    It runs ridge regression method.
    """

    print("Testing Ridge Regression...")
    w, loss = ridge_regression(y_train, x_train,lambda_)
    acc = percentage_of_correct_prediction(y_test, x_test, w)
    print("Accuracy of Ridge Regression is %2.2f out of %5d train size and %5d test size"%(acc*100,len(y_train),len(y_test)))

#-------------

def test_logistic_regression(x_train, y_train, x_test, y_test,gamma,max_iters):
    """
    It run logestic regression method.
    """
    
    print("Testing Logistic Regression...")
    initial_w = np.zeros(len(x_train[0]))
    y_train_log = y_train.copy()
    y_test_log = y_test.copy()
    y_train_log[y_train_log == -1] = 0
    y_test_log[y_test_log == -1] = 0
    w, loss = logistic_regression(y_train_log, x_train,initial_w, max_iters, gamma)
    acc = percentage_of_correct_prediction_logistic(y_test_log, x_test, w)
    print("Accuracy of Logistic Regression is %2.2f out of %5d train size and %5d test size"%(acc*100,len(y_train),len(y_test)))

#-----------

def test_logistic_ridge_regression(x_train, y_train, x_test, y_test,gamma,max_iters,lambda_):
    """
    It runs the logestic ridge regression method.
    """

    print("Testing Logistic Ridge Regression...")
    initial_w = np.zeros(len(x_train[0]))
    y_train_log = y_train.copy()
    y_test_log = y_test.copy()
    y_train_log[y_train_log == -1] = 0
    y_test_log[y_test_log == -1] = 0
    w, loss = reg_logistic_regression(y_train_log, x_train, lambda_, initial_w, max_iters, gamma)
    acc = percentage_of_correct_prediction_logistic(y_test_log, x_test, w)
    print("Accuracy of Logistic Ridge Regression is %2.2f out of %5d train size and %5d test size"%(acc*100,len(y_train),len(y_test)))

#-----------------

