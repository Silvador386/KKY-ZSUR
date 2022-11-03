import numpy as np
from utils import L2_distance_matrix

from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start = time.perf_counter()
        results = func(*args, **kwargs)
        end = time.perf_counter()
        total = end - start
        print(f"Function: {func.__name__}{args} {kwargs} Runtime: {total}")
        return results
    return timeit_wrapper


class ClassCluster:
    def __init__(self, vector):
        self.vectors = [vector]

    def add_vector(self, vector):
        self.vectors.append(vector)

    def is_vector_in_class(self, vector):
        for v in self.vectors:
            if v == vector:
                return True
        return False

    def merge_classes(self, class_b):
        self.vectors.extend(class_b.vectors)


@timeit
def cluster_level(data):
    dists_matrix = L2_distance_matrix(data, data)

    levels = []
    for i in range(dists_matrix.shape[0]-1):
        mask = np.zeros_like(dists_matrix)
        np.fill_diagonal(mask, np.infty)
        min_row, min_col = np.unravel_index(np.argmin(dists_matrix + mask), dists_matrix.shape)

        levels.append(dists_matrix[min_row][min_col])

        row_mins = np.minimum(dists_matrix[min_row, :], dists_matrix[min_col, :])
        col_mins = np.minimum(dists_matrix[:, min_row], dists_matrix[:, min_col])

        dists_matrix[min_row, :] = row_mins
        dists_matrix[:, min_row] = col_mins

        dists_matrix = np.delete(dists_matrix, min_col, 0)
        dists_matrix = np.delete(dists_matrix, min_col, 1)

        if i % 100 == 0:
            print(i)

    cls_count = 1

    levels_normed = levels / np.linalg.norm(levels)

    avg = np.average(levels_normed)
    std = np.std(levels_normed)
    value = abs(avg - std)

    for level in levels_normed:
        if level > value:
            cls_count += 1

    # print(cls_count)
    # print(avg, std)
    # print(levels_normed)

    cache = (avg, std, levels_normed)
    return cls_count, cache
