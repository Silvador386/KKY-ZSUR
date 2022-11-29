import load
import plot
import numpy as np
import clustering
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
    cls_count = 3
    # cls_count = clustering.run_cluster_methods(data_sample)
    print(f"Number of classes:{cls_count}")

    classed_data, center_data, error_cls = n_binary_division.run(data_sample, cls_count, plot=plot_option)
    print(f"Class error non-binary:  {sum(error_cls)}")
    classed_data, total_error = optimize(classed_data, center_data, plot=plot_option)
    print(f"Total error non-b opt:   {total_error}")
    classed_data, center_data, error_cls = k_means.k_means_div(data_sample, cls_count, plot=plot_option)
    print(f"Class error k-means:     {sum(error_cls)}")
    classed_data, total_error = optimize(classed_data, center_data, plot=plot_option)
    print(f"Total error k-means opt: {total_error}")
    bayes(classed_data)



if __name__ == "__main__":
    main()
