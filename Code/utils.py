import numpy as np
from functools import wraps
import time
import random


def L2_distance_matrix(A, B):
    # (X - Y)^2 = X^2 -2XY + Y^2
    dists = np.sum(A ** 2, axis=1).reshape(-1, 1) - 2*(A@B.T) + np.sum(B**2, axis=1)
    return dists


def find_cls_data_centers(data):
    center_data = []
    for cls_data in data:
        center_data.append(np.average(cls_data, axis=0))

    return center_data


def select_2params_tool(data, clustering_func):
    good_params = []
    for _ in range(5):
        NUM_SAMPLES = 500
        data_sample = data[np.random.randint(data.shape[0], size=NUM_SAMPLES)]
        dist_matrix = L2_distance_matrix(data_sample, data_sample)
        params_to_select = [[random.random()*5, random.random()*5] for _ in range(30)]
        num_cls_out = []
        for a, b in params_to_select:
            num_c, cache = clustering_func(data_sample, dist_matrix, plot=False, a=a, b=b)
            num_cls_out.append([num_c, a, b])

        num_cls_out = np.array(num_cls_out)
        mask = num_cls_out[:, 0] == 3

        good_params .extend(num_cls_out[mask][:, 1:])

    good_params = np.array(good_params)
    q_average = np.average(good_params, axis=0)
    print(q_average)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start = time.perf_counter()
        results = func(*args, **kwargs)
        end = time.perf_counter()
        total = end - start
        print(f"Function: {func.__name__} Runtime: {total}")
        return results
    return timeit_wrapper

