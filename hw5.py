import numpy as np

from scipy.optimize import minimize
from matplotlib import pyplot as plot

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


def plot_gp(data_points, prd_results):


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
    
    test_x = np.linspace(-60, 60, 120)
    prd_results = list()
    for x in test_x:
        kernel = rational_quadratic(data_points[:, 0], x, alpha, length_scale)
        variance = rational_quadratic(x, x, alpha, length_scale) + 1/beta
        mu = np.dot(kernel.T, np.dot(np.linalg.inv(C), data_points[:. 1]))
