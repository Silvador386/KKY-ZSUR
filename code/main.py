import load
import plot
import numpy as np
import clustering
import classification
import k_means
import n_binary_division
from iterative_optimization import optimize
from bayes import bayes
from utils import timeit


DATA_PATH = "../data/data.txt"


@timeit
def main():
    data = load.load_data(data_path=DATA_PATH)

    data_sample = data[np.random.randint(data.shape[0], size=800)]

    plot_option = True

    # plot.plot_2D(args)
    num_cls = 3
    # cls_count = clustering.run_cluster_methods(data_sample)
    print(f"Number of classes:{num_cls}")

    classed_data, center_data, total_error = classification.run_classification(data_sample, num_cls, plot_option)
    bayes(classed_data)


if __name__ == "__main__":
    main()
