import numpy as np
from functools import wraps
import time


def L2_distance_matrix(A, B):
    # (X - Y)^2 = X^2 -2XY + Y^2
    dists = np.sum(A ** 2, axis=1).reshape(-1, 1) - 2*(A@B.T) + np.sum(B**2, axis=1)
    return dists


def find_cls_data_centers(data):
    center_data = []
    for cls_data in data:
        center_data.append(np.average(cls_data, axis=0))

    return center_data


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

