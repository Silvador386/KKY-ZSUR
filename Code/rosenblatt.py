import numpy as np
from plot import generate_mesh, plot_mesh
from utils import timeit


@timeit
def rosenblatt(classed_data, plot=True):
    classed_data = np.copy(classed_data)
    delta = 0
    lr_constant = 1
    q_weights, num_iter, cache = train_q_weights(classed_data, delta, lr_constant)

    mesh = generate_mesh(classed_data)
    mesh_cls_idxs = predict(mesh, q_weights, cache)

    if plot:
        kwargs = {"title": "Rosenblatt classifier"}
        plot_mesh(mesh, mesh_cls_idxs, classed_data, **kwargs)

    return num_iter


def train_q_weights(classed_data, delta, lr_constant):
    num_dim = len(classed_data[0][0]) + 1
    num_classed = len(classed_data)
    cls_labels = [i for i, single_data in enumerate(classed_data) for _ in single_data]
    merged_data = np.concatenate(classed_data)
    # Transform
    merged_data, cache = transform(merged_data)

    q_size = round((num_classed * (num_classed - 1)) / 2)
    q_weights = [np.random.randint(-10, 10, size=num_dim) for _ in range(q_size)]

    num_iter = 0
    for i, q in enumerate(q_weights):
        while True:
            error = 0
            for data_point, cls_label in zip(merged_data, cls_labels):
                x = np.array([1, *data_point])
                omega = 1
                if not i == cls_label:
                    omega = -1
                condition = q.T @ x * omega

                if condition < delta:
                    q = q + lr_constant * x * omega
                    error += 1
            num_iter += 1
            if num_iter % 1000 == 0 or error == 0:
                q_weights[i] = q
                break

    return q_weights, num_iter, cache


def transform(data):
    data = np.copy(data)
    data_avg = np.average(data, axis=0)
    data -= data_avg
    cache = (data_avg)
    return data, cache


def predict(mesh, q_weights, cache):
    data_avg = cache
    mesh_cls_idxs = np.zeros(mesh.shape[0]) - 1
    for i, data_point in enumerate(mesh):
        x = np.array([1, *data_point-data_avg])
        inequalities = np.array([q.T @ x >= 0 for q in q_weights])
        if sum(inequalities) == 1:
            mesh_cls_idxs[i] = np.argwhere(inequalities)
    return mesh_cls_idxs




