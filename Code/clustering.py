import random
import numpy as np

from utils import L2_distance_matrix
from agglomerative_clustering import agglomerate_clustering
from chain_clustering import chain_clustering
from maximin_clustering import maximin_clustering


def run_clustering(data, plot):
    dists_matrix = L2_distance_matrix(data, data)

    cls_counts = []

    num_cls, cache = agglomerate_clustering(data, dists_matrix, plot)
    cls_counts.append(num_cls)
    print(f"Num_cls agg: {num_cls}, Avg: {cache[0]}, Std: {cache[1]}")

    chain_cls = []
    for i in range(6):
        num_cls, cache = chain_clustering(data, dists_matrix, random.randint(0, data.shape[0] - 1), plot)
        chain_cls.append(num_cls)
        print(f"Num_cls chain: {num_cls}, Avg: {cache[0]}, Std: {cache[1]}")
    chain_avg = round(np.average(np.array(chain_cls)))
    cls_counts.append(chain_avg)

    num_cls = maximin_clustering(data, dists_matrix)
    print(f"Num_cls maximin: {num_cls}")
    cls_counts.append(num_cls)

    avg_cls_count = round(np.average(np.array(cls_counts)))

    return avg_cls_count
