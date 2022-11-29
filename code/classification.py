import numpy as np
import k_means
import n_binary_division
from iterative_optimization import optimize
from utils import timeit


@timeit
def run_classification(data, num_cls, plot_option):
    out_data = []
    out_total_error = []

    classed_data, center_data, error_cls = n_binary_division.run(data, num_cls, plot=plot_option)
    out_data.append((classed_data, center_data))
    out_total_error.append(sum(error_cls))
    print(f"Class error non-binary:  {sum(error_cls)}")

    classed_data, center_data, total_error = optimize(classed_data, center_data, plot=plot_option)
    out_data.append((classed_data, center_data))
    out_total_error.append(total_error)
    print(f"Total error non-b opt:   {total_error}")

    classed_data, center_data, error_cls = k_means.k_means_div(data, num_cls, plot=plot_option)
    out_data.append((classed_data, center_data))
    out_total_error.append(sum(error_cls))
    print(f"Class error k-means:     {sum(error_cls)}")

    classed_data, center_data, total_error = optimize(classed_data, center_data, plot=plot_option)
    out_data.append((classed_data, center_data))
    out_total_error.append(total_error)
    print(f"Total error k-means opt: {total_error}")

    min_error_idx = np.argmin(out_total_error)
    classed_data = out_data[min_error_idx][0]
    center_data = out_data[min_error_idx][1]
    return classed_data, center_data, out_total_error[min_error_idx]
