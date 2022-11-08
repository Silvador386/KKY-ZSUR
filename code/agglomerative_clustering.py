import numpy as np
from utils import timeit
from tqdm import tqdm
from plot import plot_vector


@timeit
def cluster_level(data, dists_matrix, plot=False):
    dists_matrix = dists_matrix.copy()

    levels = []
    for i in tqdm(range(dists_matrix.shape[0]-1)):
        mask = np.zeros_like(dists_matrix)
        np.fill_diagonal(mask, np.infty)
        min_row, min_col = np.unravel_index(np.argmin(dists_matrix + mask), dists_matrix.shape)

        levels.append(dists_matrix[min_row][min_col])

        min_vector = np.minimum(dists_matrix[min_row, :], dists_matrix[min_col, :])

        dists_matrix[min_row, :] = min_vector
        dists_matrix[:, min_row] = min_vector.T

        dists_matrix = np.delete(dists_matrix, min_col, 0)
        dists_matrix = np.delete(dists_matrix, min_col, 1)

    return count_cls(levels, plot)


def count_cls(levels, plot):
    num_cls = 1
    distances = np.array(levels)
    distances_normed = distances / np.linalg.norm(distances)
    distances_normed = np.sort(distances_normed)

    step_diff = distances_normed[1:] - distances_normed[:-1]
    step_avg = np.average(step_diff)
    step_std = np.std(step_diff)

    for step in step_diff:
        if step > step_avg + 4*step_std:
            num_cls += 1

    if plot:
        plot_vector(step_diff, distances_normed)

    cache = (step_avg, step_std, distances_normed)
    return num_cls, cache
