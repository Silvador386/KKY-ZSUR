import numpy as np
import random
from utils import timeit
from plot import plot_1D


# @timeit
def chain_clustering(data, dists_matrix, plot=False, a=1, b=2):
    a, b = 2, 4
    data = np.copy(data)
    num_vectors = data.shape[0]
    dists_matrix = dists_matrix.copy()

    chain_idx_row = random.randint(0, data.shape[0] - 1)
    distances = []

    for _ in range(num_vectors):
        mask = np.zeros_like(dists_matrix)
        np.fill_diagonal(mask, np.infty)

        min_col = np.argmin((dists_matrix + mask)[chain_idx_row, :])

        distances.append(dists_matrix[chain_idx_row][min_col])

        dists_matrix = np.delete(dists_matrix, chain_idx_row, 0)
        dists_matrix = np.delete(dists_matrix, chain_idx_row, 1)

        if chain_idx_row > min_col:
            chain_idx_row = min_col
        else:
            chain_idx_row = min_col - 1

    return num_cls_step(distances, plot, a, b)


def num_cls_step(distances, plot, a, b):
    num_cls = 1
    distances = np.array(distances)

    distances = np.sort(distances)

    step_diff = distances[1:] - distances[:-1]
    step_avg = np.average(step_diff)
    step_std = np.std(step_diff)

    for step in step_diff:
        if step > (a*step_avg + b * step_std):
            num_cls += 1

    if plot:
        kwargs = {"title": "Chain clustering - distances"}
        plot_1D(distances, **kwargs)
        kwargs = {"title": "Chain clustering - step diffs"}
        plot_1D(step_diff, **kwargs)

    return num_cls
