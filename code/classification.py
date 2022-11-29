import numpy as np
from bayes import bayes


def run_classifiers(classed_data, center_data, plot_option=False):
    bayes(classed_data, center_data, plot_option)
