
import numpy as np
from aggl_clustering import cluster_level


def run_cluster_methods(data):
    num_cls, cache = cluster_level(data)
