import random

import numpy as np

from utils import L2_distance_matrix
from agglomerative_clustering import cluster_level
from chain_clustering import run_cluster
from maximin_clustering import maximin_cluster
from plot import plot_vector


def run_cluster_methods(data):
    dists_matrix = L2_distance_matrix(data, data)

    # num_cls, cache = cluster_level(data, dists_matrix)
    # print(f"Num_cls: {num_cls}, Avg: {cache[0]}, Std: {cache[1]}")
    #
    # chain_cls = []
    # for i in range(3):
    #     num_cls, cache = run_cluster(data, dists_matrix, random.randint(0, data.shape[0]))
    #     chain_cls.append(num_cls)
    #     print(f"Num_cls: {num_cls}, Avg: {cache[0]}, Std: {cache[1]}")
    # print(round(np.average(np.array(chain_cls))))

    maximin_cluster(data, dists_matrix)
