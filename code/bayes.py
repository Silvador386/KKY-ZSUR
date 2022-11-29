from plot import plot_mesh, generate_mesh
import numpy as np
from utils import timeit


@timeit
def bayes(classed_data, center_data, plot=True):
    classed_data = np.copy(classed_data)
    cls_middle_values = center_data
    cov_matrices = [cov(single_cls_data, cls_middle_value) for single_cls_data, cls_middle_value in
                    zip(classed_data, cls_middle_values)]

    mesh = generate_mesh(classed_data)
    gauss_distribs = []
    for cls_middle, cov_matrix in zip(cls_middle_values, cov_matrices):
        gauss_distribs.append([gauss_distribution(x, cls_middle, cov_matrix) for x in mesh])

    mesh_cls_idxs = np.argmax(np.vstack(gauss_distribs), axis=0)
    if plot:
        kwargs = {"title": "Bayes classifier"}
        plot_mesh(mesh, mesh_cls_idxs, classed_data, **kwargs)


def gauss_distribution(x, cls_middle, cov_matrix):
    scalar_part = 1 / (((2 * np.pi) ** (len(cls_middle) / 2)) * (np.linalg.det(cov_matrix) ** (1 / 2)))
    exp_part = (-1 / 2) * ((x - cls_middle).T @ (np.linalg.inv(cov_matrix))) @ (x - cls_middle)
    return scalar_part * np.exp(exp_part)


def cov(single_cls_data, cls_middle_value):
    cov_dim = single_cls_data.shape[1]
    cov_matrix = np.zeros((cov_dim, cov_dim))

    for i, cls_data_i in enumerate(single_cls_data.T):
        for j, cls_data_j in enumerate(single_cls_data.T):
            cov_matrix[i, j] = np.average((cls_data_i - cls_middle_value[i]) * (cls_data_j - cls_middle_value[j]).T)

    return cov_matrix
