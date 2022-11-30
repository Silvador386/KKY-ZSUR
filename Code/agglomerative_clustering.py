import numpy as np
from utils import timeit
from tqdm import tqdm
from plot import plot_1D


@timeit
def agglomerate_clustering(data, dists_matrix, plot=False, a=2, b=3):
    a, b = 2.76794805, 2.53932967
    dists_matrix = dists_matrix.copy()

    dist_levels = []
    for _ in tqdm(range(dists_matrix.shape[0]-1)):
        mask = np.zeros_like(dists_matrix)
        np.fill_diagonal(mask, np.infty)
        min_row, min_col = np.unravel_index(np.argmin(dists_matrix + mask), dists_matrix.shape)

        dist_levels.append(dists_matrix[min_row][min_col])

        min_vector = np.minimum(dists_matrix[min_row, :], dists_matrix[min_col, :])

        dists_matrix[min_row, :] = min_vector
        dists_matrix[:, min_row] = min_vector.T

        dists_matrix = np.delete(dists_matrix, min_col, 0)
        dists_matrix = np.delete(dists_matrix, min_col, 1)

    return num_cls(dist_levels, plot, a, b)


# def agg_cluster(data, dists_matrix, plot=False):
#     dists_matrix = np.copy(dists_matrix)
#     mesh = make_idxs(dists_matrix)
#     dists_matrix = np.stack([dists_matrix, mesh])
#
#     dist_levels = []
#     for _ in tqdm(range(dists_matrix.shape[0] - 1)):
#         mask = np.zeros_like(dists_matrix)
#         np.fill_diagonal(mask, np.infty)
#         min_row, min_col = np.unravel_index(np.argmin(dists_matrix + mask), dists_matrix.shape)
#
#         dist_levels.append(dists_matrix[min_row][min_col])
#
#         min_vector = np.minimum(dists_matrix[min_row, :], dists_matrix[min_col, :])
#
#         dists_matrix[min_row, :] = min_vector
#         dists_matrix[:, min_row] = min_vector.T
#
#         dists_matrix = np.delete(dists_matrix, min_col, 0)
#         dists_matrix = np.delete(dists_matrix, min_col, 1)
#
#     return num_cls(dist_levels, plot)
#
# def make_idxs(dists_matrix):
#     dists_matrix_dim = dists_matrix.shape[0]
#     col_idxs = np.arange(0, dists_matrix_dim)
#     idx_mesh = np.broadcast_to(col_idxs, dists_matrix.shape)
#     return idx_mesh


def num_cls(levels, plot, a, b):
    num_cls = 1
    distances = np.array(levels)
    distances_normed = distances / np.linalg.norm(distances)
    distances_normed = np.sort(distances_normed)

    step_diff = distances_normed[1:] - distances_normed[:-1]
    step_avg = np.average(step_diff)
    step_std = np.std(step_diff)

    for step in step_diff:
        if step > a*step_avg + b*step_std:
            num_cls += 1

    if plot:
        kwargs = {"title": "Agglomerative clustering - distances"}
        plot_1D(distances, **kwargs)
        kwargs = {"title": "Agglomerative clustering - step_diff"}
        plot_1D(step_diff, **kwargs)

    cache = (step_avg, step_std, distances_normed)
    return num_cls, cache


def select_params_tool(data):
    from utils import L2_distance_matrix
    import random

    good_params = []
    for _ in range(5):
        NUM_SAMPLES = 500
        data_sample = data[np.random.randint(data.shape[0], size=NUM_SAMPLES)]
        dist_matrix = L2_distance_matrix(data_sample, data_sample)
        params_to_select = [[random.random()*5, random.random()*5] for _ in range(20)]
        num_cls_out = []
        for a, b in params_to_select:
            num_c, cache = agglomerate_clustering(data, dist_matrix, plot=False, a=a, b=b)
            num_cls_out.append([num_c, a, b])

        num_cls_out = np.array(num_cls_out)
        mask = num_cls_out[:, 0] == 3

        good_params .extend(num_cls_out[mask][:, 1:])

    q_average = np.average(good_params, axis=0)
    print(q_average)

