import random
import numpy as np
from itertools import combinations


def maximin_clustering(data, dists_matrix, q=0.302, plot=False):
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
        max_val = mins[max_idx]

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


def select_q_tool(data):
    from utils import L2_distance_matrix

    good_q = []
    for _ in range(10):
        NUM_SAMPLES = 4000
        data_sample = data[np.random.randint(data.shape[0], size=NUM_SAMPLES)]
        dist_matrix = L2_distance_matrix(data_sample, data_sample)
        q_to_select = np.random.random_sample(50)
        num_cls_out = []
        for q in q_to_select:
            num_cls_out.append([maximin_clustering(data, dist_matrix, q), q])

        num_cls_out = np.array(num_cls_out)
        mask = num_cls_out[:, 0] == 3

        good_q .extend(num_cls_out[mask][:, 1])

    q_average = np.average(good_q)
    print(q_average)
