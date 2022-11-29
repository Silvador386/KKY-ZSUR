import numpy as np
from builtins import range
from builtins import object
from utils import L2_distance_matrix
from plot import generate_mesh, plot_mesh


def knn_classifier(classed_data, plot=True):
    classed_data = np.copy(classed_data)
    train_labels = [i for i, single_data in enumerate(classed_data) for _ in single_data]
    classifier = KNearestNeighbor()
    classifier.train(classed_data, train_labels)
    mesh = np.array(generate_mesh(classed_data))

    labels = classifier.predict(mesh, 1)
    kwargs = {"title": "knn classifier, k = 1"}
    if plot:
        plot_mesh(mesh, labels, classed_data, **kwargs)

    labels = classifier.predict(mesh, 2)
    kwargs = {"title": "knn classifier, k = 2"}
    if plot:
        plot_mesh(mesh, labels, classed_data, **kwargs)


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def train(self, classed_data, labels):
        self.X_train = np.concatenate(np.array(classed_data))
        self.y_train = np.array(labels)

    def predict(self, X, k=1):
        dist_matrix = L2_distance_matrix(X, self.X_train)

        return self.predict_labels(dist_matrix, k=k)

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # select k closest points
            closest_y = self.y_train[np.argsort(dists[i])[0:k]]

            # find the most common label in the list closest_y of labels
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred
