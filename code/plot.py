import matplotlib.pyplot as plt
import numpy as np


def plot_2D(*args, **kwargs):
    # plt.plot(args[:, 0], args[:, 1], linestyle="none", color="r", marker="o")
    if args:
        for data in args[0]:
            plt.scatter(data[:, 0], data[:, 1])
        plt.xlabel("1. column")
        plt.ylabel("2. column")
        plt.grid()
        plt.show()

    if kwargs:
        for key, data in kwargs.items():
            if key == "title":
                plt.title(data)
                continue
            plt.scatter(data[:, 0], data[:, 1], label=key)
        plt.legend()
        plt.xlabel("1. column")
        plt.ylabel("2. column")
        plt.grid()
        plt.show()


def plot_vector(*data):
    for i in data:
        plt.plot(i, scaley="log", marker="o")
    plt.show()


def generate_mesh(classed_data, num_points=100):
    merged_data = np.concatenate(classed_data)
    max_values = np.max(merged_data, axis=0)
    min_values = np.min(merged_data, axis=0)

    x = np.linspace(min_values[0]-2, max_values[0]+2, num_points)
    y = np.linspace(min_values[1] - 2, max_values[1] + 2, num_points)
    points = [[x_value, y_value] for x_value in x for y_value in y]
    return np.array(points)
