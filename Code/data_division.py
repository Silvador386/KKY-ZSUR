import numpy as np
import k_means
import n_binary_division
from iterative_optimization import optimize
from utils import timeit


@timeit
def run_division(data, num_cls, plot):
    out_data = []
    out_total_error = []

    classed_data, center_data, total_error = n_binary_division.run(data, num_cls, plot=plot)
    out_data.append((classed_data, center_data))
    out_total_error.append(total_error)
    print(f"Class error non-binary:  {total_error}")

    classed_data, center_data, total_error = optimize(classed_data, center_data, plot=plot)
    out_data.append((classed_data, center_data))
    out_total_error.append(total_error)
    print(f"Total error non-b opt:   {total_error}")

    classed_data, center_data, total_error = k_means.k_means_div(data, num_cls, plot=plot)
    out_data.append((classed_data, center_data))
    out_total_error.append(total_error)
    print(f"Class error k-means:     {total_error}")

    classed_data, center_data, total_error = optimize(classed_data, center_data, plot=plot)
    out_data.append((classed_data, center_data))
    out_total_error.append(total_error)
    print(f"Total error k-means opt: {total_error}")

    run_further(data, num_cls, out_data, out_total_error)

    min_error_idx = np.argmin(out_total_error)
    classed_data = out_data[min_error_idx][0]
    center_data = out_data[min_error_idx][1]
    print(f"Final minimal error: {out_total_error[min_error_idx]}")
    return np.array(classed_data, dtype=object), center_data, out_total_error[min_error_idx]


def run_further(data, num_cls, out_data, out_total_error, plot=False):
    for _ in range(3):
        classed_data, center_data, total_error = n_binary_division.run(data, num_cls, plot=plot)
        out_data.append((classed_data, center_data))
        out_total_error.append(total_error)

        classed_data, center_data, total_error = optimize(classed_data, center_data, plot=plot)
        out_data.append((classed_data, center_data))
        out_total_error.append(total_error)

        classed_data, center_data, total_error = k_means.k_means_div(data, num_cls, plot=plot)
        out_data.append((classed_data, center_data))
        out_total_error.append(total_error)

        classed_data, center_data, total_error = optimize(classed_data, center_data, plot=plot)
        out_data.append((classed_data, center_data))
        out_total_error.append(total_error)
