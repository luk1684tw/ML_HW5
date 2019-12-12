import numpy as np

from scipy.optimize import minimize
from matplotlib import pyplot as plt

def extract_gp_data():
    data = open('./input.data')
    data_points = list()
    for data_point in data:
        data_point = [float(i) for i in data_point.split()]
        data_points.append(data_point)

    data_points = np.array(data_points)
    return (data_points)


def rational_quadratic(x, y, alpha, length_scale):
    return (1 + (x - y)**2 / (2 * alpha * length_scale**2)) ** (-1 * alpha)


def plot_gp(data_points, test_x, prd_results, fig_name):
    plt.plot(data_points[:, 0], data_points[:, 1], 'ro')
    plt.fill_between(x=test_x, y1=prd_results[:, 0] + 1.96*np.sqrt(prd_results[:, 1]), y2=prd_results[:, 0] - 1.96*np.sqrt(prd_results[:, 1]))
    plt.plot(test_x, prd_results[:, 0], 'r--')
    plt.savefig(f'{fig_name}.png')
    return

if __name__ == "__main__":
    data_points = extract_gp_data()

    # 1 Gaussian Process
    beta = 5
    C = list()
    alpha = 1
    length_scale = 1

    for i in range(34):
        C_i = rational_quadratic(data_points[i, 0], data_points[:, 0], alpha, length_scale)
        C.append(C_i)
    C = np.array(C)
    C += np.identity(len(C)) / beta
    
    test_x = np.linspace(-60, 60, 240)
    prd_results = list()
    for x in test_x:
        kernel = rational_quadratic(data_points[:, 0], x, alpha, length_scale).reshape((34, 1))
        variance = rational_quadratic(x, x, alpha, length_scale) + 1/beta
        mu = np.dot(kernel.T, np.dot(np.linalg.inv(C), data_points[:, 1]))
        var = variance - np.dot(kernel.T, np.dot(np.linalg.inv(C), kernel))
        var = var.reshape((1))
        res = np.concatenate((mu, var), axis=0)
        prd_results.append(res)
    prd_results = np.array(prd_results)
    plot_gp(data_points, test_x, prd_results, 'GP_origin')

    
