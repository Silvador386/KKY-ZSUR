import numpy as np
from functools import wraps
import time
from plot import plot_vector


def count_cls(distances):
    num_cls = 1
    distances = np.array(distances)
    distances_normed = distances / np.linalg.norm(distances)
    avg = np.average(distances_normed)
    std = np.std(distances_normed)

    distances_normed = np.sort(distances_normed)

    step_diff = distances_normed[1:] - distances_normed[:-1]
    step_avg = np.average(step_diff)
    step_std = np.std(step_diff)

    for step in step_diff:
        if step > (3*step_avg + 5 * step_std):
            num_cls += 1

    # plot_vector(step_diff, distances_normed)
    # print(step_avg, step_std, step_diff[-10:-1])

    cache = (step_avg, step_std, distances_normed)
    return num_cls, cache



def L2_distance_matrix(A, B):
    # # (x - y)^2 = x^2 + y^2 -2xy


    # X = A  # test args (m, d)
    # X_train = A  # train args (n, d)

    # this has the same affect as taking the dot product of each row with itself
    # x2 = np.sum(X ** 2, axis=1)  # shape of (m)
    # y2 = np.sum(X_train ** 2, axis=1)  # shape of (n)

    # we can compute all x_i * y_j and store it in a matrix at xy[i][j] by
    # taking the matrix multiplication between X and X_train transpose
    # if you're stuggling to understand this, draw out the matrices and
    # do the matrix multiplication by hand
    # (m, d) x (d, n) -> (m, n)
    # xy = np.matmul(X, X_train.T)

    # each row in xy needs to be added with x2[i]
    # each column of xy needs to be added with y2[j]
    # to get everything to play well, we'll need to reshape
    # x2 from (m) -> (m, 1), numpy will handle the rest of the broadcasting for us
    # see: https://numpy.org/doc/stable/user/basics.broadcasting.html
    # x2 = x2.reshape(-1, 1)
    # dists = x2 - 2 * xy + y2  # (m, 1) repeat columnwise + (m, n) + (n) repeat rowwise -> (m, n)

    # (X - Y)
    dists = np.sum(A ** 2, axis=1).reshape(-1, 1) - 2*(A@B.T) + np.sum(B**2, axis=1)

    return dists


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


def prep_data(dists_matrix):
    upper_dist_list = []

    triu_idxs = np.triu_indices_from(dists_matrix, k=1)

    data = dists_matrix[triu_idxs]

    for dist, idx1, idx2 in zip(data, *triu_idxs):
        upper_dist_list.append([dist, idx1, idx2])

    upper_dist_list = np.array(upper_dist_list)
    upper_dist_list = upper_dist_list[upper_dist_list[:, 0].argsort()]

    return upper_dist_list


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