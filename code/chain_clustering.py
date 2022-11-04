import numpy as np
from utils import timeit, count_cls
from tqdm import tqdm

from plot import plot_vector


@timeit
def run_cluster(data, dists_matrix, chain_idx_row=0):
    num_vectors = data.shape[0]
    dists_matrix = dists_matrix.copy()

    distances = []

    for i in tqdm(range(num_vectors)):
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

    num_cls = 1
    distances = np.array(distances)
    distances_normed = distances / np.linalg.norm(distances)
    avg = np.average(distances_normed)
    std = np.std(distances_normed)

    distances_normed = np.sort(distances_normed)

    step_diff = distances_normed[1:] - distances_normed[:-1]
    step_avg = np.average(step_diff)
    step_std = np.std(step_diff)

    for step in step_diff:
        if step > (step_avg + 3 * step_std):
            num_cls += 1

    plot_vector(step_diff, distances_normed)
    # print(step_avg, step_std, step_diff[-10:-1])

    cache = (avg, std, distances_normed)
    return num_cls, cache


