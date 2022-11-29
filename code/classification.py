import numpy as np
from bayes import bayes
from vector_quantization import vector_quantization
from k_nearest_neighbour import knn_classifier
from rosenblatt import rosenblatt


def run_classifiers(classed_data, center_data, plot_option=False):
    # bayes(classed_data, center_data, plot_option)
    # vector_quantization(classed_data, center_data, plot_option)
    # knn_classifier(classed_data)
    rosenblatt(classed_data)
