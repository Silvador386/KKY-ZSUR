import numpy as np
from functools import wraps
import time
from plot import plot_vector


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

