import random

import numpy as np
from itertools import combinations


def maximin_cluster(data, dists_matrix):
    dists_matrix = dists_matrix.copy()
    q = 0.2
    center_idxs = []

    start_idx = random.randint(0, dists_matrix.shape[0])
    center_idxs.append(start_idx)

    end_idx = np.argmax(dists_matrix[start_idx, :])
    center_idxs.append(end_idx)

    while True:

        dist_vectors = []

        for center in center_idxs:
            # mask = np.ones((dists_matrix.shape[0]), bool)
            # mask[center] = False
            max_center_dist = dists_matrix[center, :]
            dist_vectors.append(max_center_dist)

        dist_vectors = np.array(dist_vectors)
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
        else:
            break

    num_cls = len(center_idxs)

    print(num_cls)
