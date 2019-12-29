import numpy as np


from scipy.optimize import minimize
from matplotlib import pyplot as plt


from libsvm.python.svmutil import *
from libsvm.python.svm import *

import csv
from scipy.spatial.distance import cdist

def extract_gp_data():
    data = open('./input.data')
    data_points = list()
    for data_point in data:
        data_point = [float(i) for i in data_point.split()]
        data_points.append(data_point)

    data_points = np.array(data_points)
    return (data_points)


def make_C(data_points, alpha, length_scale, sigma, beta):
    C = list()
    for i in range(34):
        C_i = rational_quadratic(data_points[i, 0], data_points[:, 0], alpha, length_scale, sigma)
        C.append(C_i)
    C = np.array(C)
    C += np.identity(len(C)) / beta

    return C


def rational_quadratic(x, y, alpha, length_scale, sigma):

    return sigma**2 * (1 + (y - x)**2 / (2 * alpha * length_scale**2)) ** (-1 * alpha)


def optimize(theta, data_points, beta):
    C = make_C(data_points, theta[0], theta[1], theta[2], beta)
    return 0.5 * np.log(np.linalg.det(C)) + 0.5 * np.dot(data_points[:, 1].T, np.dot(np.linalg.inv(C), data_points[:, 1])) + 0.5*len(data_points)*np.log(2*np.pi)


def plot_gp(data_points, test_x, prd_results, fig_name):
    plt.plot(data_points[:, 0], data_points[:, 1], 'bo')
    plt.fill_between(x=test_x, y1=prd_results[:, 0] + 1.96*np.sqrt(prd_results[:, 1]), y2=prd_results[:, 0] - 1.96*np.sqrt(prd_results[:, 1]), color='pink')
    plt.plot(test_x, prd_results[:, 0], linestyle='-', color='red')
    plt.savefig(f'{fig_name}.png')
    return


def gp_predict(test_x, data_points, alpha, length_scale, sigma, beta):
    prd_results = list()
    for x in test_x:
        kernel = rational_quadratic(data_points[:, 0], x, alpha, length_scale, sigma).reshape((34, 1))
        variance = rational_quadratic(x, x, alpha, length_scale, sigma) + 1/beta
        mu = np.dot(kernel.T, np.dot(np.linalg.inv(C), data_points[:, 1]))
        var = variance - np.dot(kernel.T, np.dot(np.linalg.inv(C), kernel))

        var = var.reshape((1))
        res = np.concatenate((mu, var), axis=0)
        prd_results.append(res)
    prd_results = np.array(prd_results)
    return prd_results


def libsvm_input(data):
    data_list = list()
    data = csv.reader(data)
    for img in data:
        img_list = list()
        for pixel in img:
            img_list.append(float(pixel))
        data_list.append(img_list)
    
    return data_list


def rbf(x, y, gamma):
    val = cdist(x, y, metric='euclidean')**2
    return np.exp(-gamma * val)


def linear_rbf(data, gamma):
    linear = np.dot(data, data.T)
    n = data.shape[0]
    kernel = np.zeros((n, n+1))
    kernel[:, 1:] = rbf(data, data, gamma) + linear
    kernel[:, 0] = np.arange(1, n + 1)
    return kernel
    

if __name__ == "__main__":
    # data_points = extract_gp_data()

    # # 1 Gaussian Process
    # beta = 5
    # alpha = 1
    # length_scale = 1
    # sigma = 1
    # C = make_C(data_points, alpha, length_scale, sigma, beta)
    # # print (C)
    
    # test_x = np.linspace(-60, 60, 1000)
    # prd_results = gp_predict(test_x, data_points, alpha, length_scale, sigma, beta)
    # plot_gp(data_points, test_x, prd_results, 'GP_origin')
    # plt.close()

    # theta = np.array([alpha, length_scale, sigma])

    # result = minimize(optimize, theta, args=(data_points, beta), method = 'Nelder-Mead')
    # print (result)
    # alpha, length_scale, sigma = result.x[0], result.x[1], result.x[2]
    # C = make_C(data_points, alpha, length_scale, sigma, beta)

    # prd_results = gp_predict(test_x, data_points, alpha, length_scale, sigma, beta)
    # plot_gp(data_points, test_x, prd_results, 'GP_finetuned')
    # plt.close()

    # 2 SVM
    train_x = np.genfromtxt("./X_train.csv", delimiter=',').astype(float)
    train_y = np.genfromtxt("./Y_train.csv", delimiter=',').astype(int)
    test_x = np.genfromtxt("./X_test.csv", delimiter=',').astype(float)
    test_y = np.genfromtxt("./Y_test.csv", delimiter=',').astype(int)
    # with open('mnist/Y_train.csv', 'rt') as data:
    #     tmp_train_y = list(csv.reader(data))
    # with open ('mnist/Y_test.csv', 'rt') as data:
    #     tmp_test_y = list(csv.reader(data))

    # train_y, test_y = list(), list()
    # for label in tmp_train_y:
    #     train_y.append(int(label[0]))
    # for label in tmp_test_y:
    #     test_y.append(int(label[0]))

    # train_y, test_y = np.array(train_y), np.array(test_y)
    # train_x, test_x = np.array(train_x), np.array(test_x)
    # print (train_x.shape)
    print ('start computing custom kernel')
    custom_train_x = linear_rbf(train_x, 1/32)
    custom_test_x = linear_rbf(test_x, 1/32)
    print ('end of computing custom kernel')

    # model_linear = svm_train(train_y, train_x, '-t 0 -q')
    # model_poly = svm_train(train_y, train_x, '-t 1 -q')
    # model_rbf = svm_train(train_y, train_x, '-t 2 -q')
    model_linear_rbf = svm_train(train_y, custom_train_x, '-t 4 -q')

    
    # prd_linear = svm_predict(test_y, test_x, model_linear)
    # prd_poly = svm_predict(test_y, test_x, model_poly)
    # prd_rbf = svm_predict(test_y, test_x, model_rbf)
    prd_linear_rbf = svm_predict(test_y, custom_test_x, model_linear_rbf)

    
    print ('--------------------------------')
    # print (prd_linear)
    # print (prd_poly)
    # print ('--------------------------------')
    # print (prd_rbf)