from utils import L2_distance_matrix
from plot import plot_2D, generate_mesh
import matplotlib.pyplot as plt
import numpy as np


def bayes(classed_data, center_data, plot=True):
    classed_data = np.copy(classed_data)
    cls_middle_values = center_data
    cov_matrices = [cov(single_cls_data, cls_middle_value) for single_cls_data, cls_middle_value in
                     zip(classed_data, cls_middle_values)]

    mesh = generate_mesh(classed_data)
    gauss_distribs = []
    for cls_middle, cov_matrix in zip(cls_middle_values, cov_matrices):
        gauss_distribs.append([gauss_distribution(x, cls_middle, cov_matrix) for x in mesh])

    surface_cls = np.argmax(np.vstack(gauss_distribs), axis=0)
    if plot:
        plot_mesh(mesh, surface_cls, classed_data)


def gauss_distribution(x, cls_middle, cov_matrix):
    scalar_part = 1 / (((2 * np.pi) ** (len(cls_middle) / 2)) * (np.linalg.det(cov_matrix) ** (1 / 2)))
    exp_part = (-1 / 2) * ((x - cls_middle).T @ (np.linalg.inv(cov_matrix))) @ (x - cls_middle)
    return scalar_part * np.exp(exp_part)


def plot_mesh(mesh_data, surface_cls, classed_data):
    data_to_scatter_plot = []
    for i in range(np.max(surface_cls)+1):
        mask = surface_cls == i
        sub_data = mesh_data[mask]
        data_to_scatter_plot.append(sub_data)
    data_to_scatter_plot += classed_data.tolist()
    plot_2D(data_to_scatter_plot)


def cov(single_cls_data, cls_middle_value):
    cov_dim = single_cls_data.shape[1]

    cov_matrix = np.zeros((cov_dim, cov_dim))

    for i, cls_data_i in enumerate(single_cls_data.T):
        for j, cls_data_j in enumerate(single_cls_data.T):
            cov_matrix[i, j] = np.average((cls_data_i - cls_middle_value[i]) * (cls_data_j - cls_middle_value[j]).T)

    return cov_matrix
