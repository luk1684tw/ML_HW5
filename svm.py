from libsvm.python.svmutil import *
from libsvm.python.svm import *

import csv
from scipy.spatial.distance import cdist

import numpy as np

import time


def rbf(x, y, gamma):
    val = cdist(x, y, metric='euclidean')**2
    return np.exp(-gamma * val)


def linear_rbf(data, gamma):
    linear = np.dot(data, data.T)
    kernel = np.zeros((len(data), len(data)+1))
    kernel[:, 1:] = rbf(data, data, gamma) + linear
    kernel[:, 0] = np.arange(1, len(data) + 1)
    return kernel
    

def grid_search(method):
    paras = {'g': [0.25, 0.5, 1, 2, 4], 'c': [0.25, 0.5, 1, 2, 4]}
    best_paras = (0, 1, 1)
    for g in paras['g']:
        for c in paras['c']:
            if method == 'linear':
                model = svm_train(train_y, train_x, f'-t 0 -q -g {g} -c {c} -v 5')        
            elif method == 'poly':
                model = svm_train(train_y, train_x, f'-t 1 -q -g {g} -c {c} -v 5')        
            elif method == 'rbf':
                model = svm_train(train_y, train_x, f'-t 2 -q -g {g} -c {c} -v 5')
            else:
                model = svm_train(train_y, custom_train_x, f'-t 4 -q -g {g} -c {c} -v 5')
            if model > best_paras[0]:
                best_paras = (model, g, c)

    print (f'best score for {method}', best_paras)
    return best_paras

if __name__ == "__main__":

    train_x = np.genfromtxt("./X_train.csv", delimiter=',').astype(float)
    train_y = np.genfromtxt("./Y_train.csv", delimiter=',').astype(int)
    test_x = np.genfromtxt("./X_test.csv", delimiter=',').astype(float)
    test_y = np.genfromtxt("./Y_test.csv", delimiter=',').astype(int)

    # print ('start computing custom kernel')
    # custom_train_x = linear_rbf(train_x, 1/32)
    # custom_test_x = linear_rbf(test_x, 1/32)
    # print ('end of computing custom kernel')

    start = time.time()
    model_linear = svm_train(train_y, train_x, '-t 0 -q')
    print ('Linear model training use', time.time() - start , 'seconds')
    start = time.time()
    model_poly = svm_train(train_y, train_x, '-t 1 -q')
    print ('Poly model training use', time.time() - start , 'seconds')
    start = time.time()
    model_rbf = svm_train(train_y, train_x, '-t 2 -q')
    print ('RBF model training use', time.time() - start , 'seconds')
    # start = time.time()
    # model_linear_rbf = svm_train(train_y, custom_train_x, '-t 4 -q')
    # print ('Custom model training use', time.time() - start , 'seconds')


    prd_linear = svm_predict(test_y, test_x, model_linear)
    prd_poly = svm_predict(test_y, test_x, model_poly)
    prd_rbf = svm_predict(test_y, test_x, model_rbf)
    # prd_linear_rbf = svm_predict(test_y, custom_test_x, model_linear_rbf)

    
    # print ('--------------------------------')
    grid_linear = grid_search('linear')
    grid_poly = grid_search('poly')
    grid_rbf = grid_search('rbf')
    model_linear = svm_train(train_y, train_x,  f'-t 0 -q -g {grid_linear[1]} -c {grid_linear[2]}')
    model_poly = svm_train(train_y, train_x,  f'-t 0 -q -g {grid_poly[1]} -c {grid_poly[2]}')
    model_rbf = svm_train(train_y, train_x,  f'-t 0 -q -g {grid_rbf[1]} -c {grid_rbf[2]}')

    prd_linear = svm_predict(test_y, test_x, model_linear)
    prd_poly = svm_predict(test_y, test_x, model_poly)
    prd_rbf = svm_predict(test_y, test_x, model_rbf)