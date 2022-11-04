import load
import plot
import numpy as np
import clustering


DATA_PATH = "../data/data.txt"


def main():
    data = load.load_data(data_path=DATA_PATH)

    data_sample = data[np.random.randint(data.shape[0], size=3000)]



    # plot.plot_2D(data)
    clustering.run_cluster_methods(data_sample)


if __name__ == "__main__":
    main()
