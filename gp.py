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

if __name__ == "__main__":
    data_points = extract_gp_data()

    # 1 Gaussian Process
    beta = 5
    alpha = 1
    length_scale = 1
    sigma = 1
    C = make_C(data_points, alpha, length_scale, sigma, beta)
    # print (C)
    
    test_x = np.linspace(-60, 60, 1000)
    prd_results = gp_predict(test_x, data_points, alpha, length_scale, sigma, beta)
    plot_gp(data_points, test_x, prd_results, 'GP_origin')
    plt.close()

    theta = np.array([alpha, length_scale, sigma])

    result = minimize(optimize, theta, args=(data_points, beta), method = 'Nelder-Mead')
    print (result)
    alpha, length_scale, sigma = result.x[0], result.x[1], result.x[2]
    C = make_C(data_points, alpha, length_scale, sigma, beta)

    prd_results = gp_predict(test_x, data_points, alpha, length_scale, sigma, beta)
    plot_gp(data_points, test_x, prd_results, 'GP_finetuned')
    plt.close()
