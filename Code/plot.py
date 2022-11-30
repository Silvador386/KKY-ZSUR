import matplotlib.pyplot as plt
import numpy as np


def plot_2D(*args, **kwargs):
    # plt.plot(args[:, 0], args[:, 1], linestyle="none", color="r", marker="o")
    if args:
        if kwargs:
            for key, data in kwargs.items():
                if key == "title":
                    plt.title(data)

        for data in args[0]:
            plt.scatter(data[:, 0], data[:, 1])
        plt.xlabel("1. column")
        plt.ylabel("2. column")
        plt.grid()
        plt.show()

    elif kwargs:
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


def plot_1D(*args, **kwargs):
    if args:
        if kwargs:
            for key, data in kwargs.items():
                if key == "title":
                    plt.title(data)

        for data in args:
            plt.plot(data[:], linestyle="none", color="r", marker="o")
        plt.xlabel("Step")
        plt.ylabel("Data distances")
        plt.grid()
        plt.show()


def plot_mesh(mesh_data, mesh_cls_idxs, classed_data, **kwargs):
    data_to_scatter_plot = []
    for i in range(int(np.max(mesh_cls_idxs)+1)):
        mask = mesh_cls_idxs == i
        sub_data = mesh_data[mask]
        data_to_scatter_plot.append(sub_data)
    data_to_scatter_plot += classed_data.tolist()
    plot_2D(data_to_scatter_plot, **kwargs)


def generate_mesh(classed_data, num_points=100):
    merged_data = np.concatenate(classed_data)
    max_values = np.max(merged_data, axis=0)
    min_values = np.min(merged_data, axis=0)

    x = np.linspace(min_values[0]-2, max_values[0]+2, num_points)
    y = np.linspace(min_values[1] - 2, max_values[1] + 2, num_points)
    points = [[x_value, y_value] for x_value in x for y_value in y]
    return np.array(points)
