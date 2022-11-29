from utils import L2_distance_matrix
import matplotlib.pyplot as plt
import numpy as np


def bayes(classed_data):
    classed_data = np.copy(classed_data)
    num_cls = len(classed_data)
    cls_num_samples = [len(single_cls_data) for single_cls_data in classed_data]
    cls_middle_values = [np.average(single_cls_data, axis=0) for single_cls_data in classed_data]
    cov_matrices = [cov(single_cls_data, cls_middle_value) for single_cls_data, cls_middle_value in
                     zip(classed_data, cls_middle_values)]

    # cov_matrix2 = [np.cov(single_cls_data.T) for single_cls_data in classed_data]
    mesh = generate_mesh(classed_data)
    gauss_distribs = []
    for cls_middle, cov_matrix in zip(cls_middle_values, cov_matrices):
        gauss_distribs.append([gauss_distribution(x, cls_middle, cov_matrix) for x in mesh])

    surface_cls = np.argmax(np.vstack(gauss_distribs), axis=0)
    for i in range(3):
        mask = surface_cls == i
        sub_data = mesh[mask]
        plt.scatter(sub_data[:, 0], sub_data[:, 1])

    for single_class_data in classed_data:
        plt.scatter(single_class_data[:, 0], single_class_data[:, 1])
    plt.show()


def generate_mesh(classed_data, num_points=100):
    merged_data = np.concatenate(classed_data)
    max_values = np.max(merged_data, axis=0)
    min_values = np.min(merged_data, axis=0)

    x = np.linspace(min_values[0]-2, max_values[0]+2, num_points)
    y = np.linspace(min_values[1] - 2, max_values[1] + 2, num_points)
    points = [[x_value, y_value] for x_value in x for y_value in y]
    return np.array(points)


def gauss_distribution(x, cls_middle, cov_matrix):
    scalar_part = 1 / (((2 * np.pi) ** (len(cls_middle) / 2)) * (np.linalg.det(cov_matrix) ** (1 / 2)))
    exp_part = (-1 / 2) * ((x - cls_middle).T @ (np.linalg.inv(cov_matrix))) @ (x - cls_middle)
    return scalar_part * np.exp(exp_part)


def plot_mesh(mesh, gauss_distribs, classed_data):
    pass




def cov(single_cls_data, cls_middle_value):
    cov_dim = single_cls_data.shape[1]

    cov_matrix = np.zeros((cov_dim, cov_dim))

    for i, cls_data_i in enumerate(single_cls_data.T):
        for j, cls_data_j in enumerate(single_cls_data.T):
            cov_matrix[i, j] = np.average((cls_data_i - cls_middle_value[i]) * (cls_data_j - cls_middle_value[j]).T)

    return cov_matrix
