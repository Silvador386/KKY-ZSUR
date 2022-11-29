import load
import plot
import numpy as np
import clustering
import data_division
import classification
from utils import timeit


DATA_PATH = "../data/data.txt"


@timeit
def main():
    data = load.load_data(data_path=DATA_PATH)

    data_sample = data[np.random.randint(data.shape[0], size=500)]

    plot_option = True

    # plot.plot_2D(args)
    num_cls = 3
    # num_cls = clustering.run_cluster_methods(data_sample)
    print(f"Number of classes:{num_cls}")

    classed_data, center_data, total_error = data_division.run_classification(data_sample, num_cls, plot_option)
    classification.run_classifiers(classed_data, center_data, plot_option)


if __name__ == "__main__":
    main()
