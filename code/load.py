import numpy as np


def load_data(data_path):
    with open(data_path, "r") as fp:
        lines = []
        for line in fp:
            lines.append([float(value) for value in line.split()])
    return np.array(lines)
