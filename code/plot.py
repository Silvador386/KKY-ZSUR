import matplotlib.pyplot as plt


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