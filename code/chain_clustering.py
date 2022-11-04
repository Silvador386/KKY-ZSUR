import numpy as np
from utils import timeit, count_cls
from tqdm import tqdm

from plot import plot_vector


@timeit
def run_cluster(data, dists_matrix, chain_idx_row=0):
    num_vectors = data.shape[0]
    dists_matrix = dists_matrix.copy()

    distances = []

    chain_idx_row = 0
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

    return count_cls(distances)


