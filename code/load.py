import numpy as np


def load_data(data_path):
    with open(data_path, "r") as fp:
        lines = fp.readlines()
        np_lines = np.array([[float(value) for value in line.split()] for line in lines])

    return np_lines
