import load
import plot
import numpy as np
import clustering
import k_means
import n_binary_division
from iterative_optimization import optimize


DATA_PATH = "../data/data.txt"


def main():
    data = load.load_data(data_path=DATA_PATH)

    data_sample = data[np.random.randint(data.shape[0], size=500)]

    # plot.plot_2D(args)
    cls_count = 3
    cls_count = clustering.run_cluster_methods(data_sample)
    print(f"Number of classes:{cls_count}")

    n_binary_division.run(data_sample, cls_count, plot=True)
    clustered_data, center_data = k_means.k_means_div(data_sample, cls_count, plot=True)
    # optimize(clustered_data, center_data)


if __name__ == "__main__":
    main()
