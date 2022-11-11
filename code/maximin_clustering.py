import math
import random
import numpy as np
from itertools import combinations


from utils import timeit


@timeit
def maximin_cluster(data, dists_matrix, q=0.314):
    dists_matrix = dists_matrix.copy()
    center_idxs = []

    start_idx = random.randint(0, dists_matrix.shape[0]-1)
    center_idxs.append(start_idx)

    end_idx = np.argmax(dists_matrix[start_idx, :])
    center_idxs.append(end_idx)

    dist_vectors = []

    for center in center_idxs:
        max_center_dist = dists_matrix[center, :]
        dist_vectors.append(max_center_dist)

    dist_vectors = np.array(dist_vectors)

    while True:
        mins = np.min(dist_vectors, axis=0)

        max_idx = np.argmax(mins)
        max_val = np.max(mins)

        combs = combinations(center_idxs, 2)

        total = 0
        length = 0
        for row, col in combs:
            total += dists_matrix[row, col]
            length += 1

        if max_val > q * total / length:
            center_idxs.append(max_idx)
            dist_vectors = np.vstack([dist_vectors, dists_matrix[max_idx, :]])
        else:
            break

    num_cls = len(center_idxs)

    return num_cls
