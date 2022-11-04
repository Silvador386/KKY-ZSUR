import numpy as np
from utils import timeit, count_cls
from tqdm import tqdm


@timeit
def cluster_level(data, dists_matrix):
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

    # cls_count = 1
    #
    # levels_normed = levels / np.linalg.norm(levels)
    #
    # avg = np.average(levels_normed)
    # std = np.std(levels_normed)
    # value = abs(avg - std)
    #
    # for i, level in enumerate(levels_normed[:-1]):
    #     if levels_normed[i+1] - level > 0.15:
    #         cls_count += 1
    #
    # cache = (avg, std, levels_normed)
    # return cls_count, cache

    return count_cls(levels)
