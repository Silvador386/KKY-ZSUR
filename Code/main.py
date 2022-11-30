import load
import numpy as np
import clustering
import data_division
import classification
from nn import neural_network
from utils import timeit
from maximin_clustering import select_q_tool
from agglomerative_clustering import select_params_tool


DATA_PATH = "../data/data.txt"
NUM_SAMPLES = 1200
PLOT = True


@timeit
def main():
    # Load data
    data = load.load_data(data_path=DATA_PATH)
    data_sample = data[np.random.randint(data.shape[0], size=NUM_SAMPLES)]

    # select_q_tool(data)
    # select_params_tool(data)

    # Clustering - Predict the number of classes.
    num_cls = 3
    num_cls = clustering.run_clustering(data_sample, PLOT)
    print(f"Number of classes:{num_cls}")

    # Division - Divide data point to classes.
    classed_data, center_data, total_error = data_division.run_division(data_sample, num_cls, PLOT)

    # Classification - Run classifiers on the divided (classed) data.
    classification.run_classifiers(classed_data, center_data, PLOT)

    # Neural network - Run 2-layer net on the classed data.
    neural_network.run(classed_data, num_cls)


if __name__ == "__main__":
    main()
