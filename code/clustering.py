import numpy as np

from utils import L2_distance_matrix
from agglomerative_clustering import cluster_level
from chain_clustering import run_cluster
from plot import plot_vector


def run_cluster_methods(data):
    dists_matrix = L2_distance_matrix(data, data)

    num_cls, cache = cluster_level(data, dists_matrix)
    print(f"Num_cls: {num_cls}, Avg: {cache[0]}, Std: {cache[1]}")

    num_cls, cache = run_cluster(data, dists_matrix)
    print(f"Num_cls: {num_cls}, Avg: {cache[0]}, Std: {cache[1]}")


