import matplotlib.pyplot as plt


def plot_2D(data):
    plt.plot(data[:, 0], data[:, 1], linestyle="none", color="r", marker="o")
    plt.xlabel("1. column")
    plt.ylabel("2. column")
    plt.show()
