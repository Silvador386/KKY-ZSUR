import load
import plot
import numpy as np
import clustering
import k_means


DATA_PATH = "../data/data.txt"


def main():
    data = load.load_data(data_path=DATA_PATH)

    data_sample = data[np.random.randint(data.shape[0], size=500)]

    # plot.plot_2D(data)
    cls_count = clustering.run_cluster_methods(data_sample)
    print(f"Number of classes:{cls_count}")
    k_means.k_means_div(data_sample, cls_count)


if __name__ == "__main__":
    main()
