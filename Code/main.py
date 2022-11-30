import load
import numpy as np
import clustering
import data_division
import classification
from nn import neural_network
from utils import timeit


DATA_PATH = "../data/data.txt"
NUM_SAMPLES = 800
PLOT = True


@timeit
def main():
    # Load data
    data = load.load_data(data_path=DATA_PATH)
    data_sample = data[np.random.randint(data.shape[0], size=NUM_SAMPLES)]

    # Clustering - Predict the number of classes.
    num_cls = 3
    # num_cls = clustering.run_clustering(data_sample)
    print(f"Number of classes:{num_cls}")

    # Division - Divide data point to classes.
    classed_data, center_data, total_error = data_division.run_division(data_sample, num_cls, PLOT)

    # Classification - Run classifiers on the divided (classed) data.
    classification.run_classifiers(classed_data, center_data, PLOT)

    # Neural network - Run 2-layer net on the classed data.
    neural_network.run(classed_data, num_cls)


if __name__ == "__main__":
    main()
