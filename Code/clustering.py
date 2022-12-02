import random
import numpy as np
from tqdm import tqdm

from utils import L2_distance_matrix
from agglomerative_clustering import agglomerate_clustering
from chain_clustering import chain_clustering
from maximin_clustering import maximin_clustering
from utils import timeit


@timeit
def run_clustering(data, plot):
    cls_counts = []

    agg_avg = run_clustering_func_ntimes(data, agglomerate_clustering, plot_last=plot, num_times=3, data_size=800)
    print(f"Agg num cls avg:     {agg_avg} -> {round(agg_avg)}")
    cls_counts.append(round(agg_avg))

    chain_avg = run_clustering_func_ntimes(data, chain_clustering, plot_last=plot, num_times=100, data_size=300)
    print(f"Chain num cls avg:   {chain_avg} -> {round(chain_avg)}")
    cls_counts.append(round(chain_avg))

    maximin_avg = run_clustering_func_ntimes(data, maximin_clustering, plot_last=plot, num_times=100, data_size=300)
    print(f"Maximin num cls avg: {maximin_avg} -> {round(maximin_avg)}")
    cls_counts.append(round(maximin_avg))

    avg_cls_count = round(np.average(np.array(cls_counts)))

    return avg_cls_count


def run_clustering_func_ntimes(data, func, plot_last, num_times, data_size=300):
    num_cls_list = []
    for i in tqdm(range(num_times)):
        data_sample = data[np.random.choice(data.shape[0], size=min(data_size, data.shape[0]), replace=False), :]
        dists_matrix = L2_distance_matrix(data_sample, data_sample)
        if plot_last and i == num_times-1:
            num_cls_list.append(func(data_sample, dists_matrix, plot=True))
        else:
            num_cls_list.append(func(data_sample, dists_matrix, plot=False))

    num_cls = np.average(num_cls_list)
    return num_cls
