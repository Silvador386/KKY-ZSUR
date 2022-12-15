import numpy as np
from plot import generate_mesh, plot_mesh
from rosenblatt import predict, transform
from utils import timeit


@timeit
def constant_increase(classed_data, plot=True):
    classed_data = np.copy(classed_data)
    delta = 0
    beta = 0.1
    q_weights, num_iter, cache = train_q_weights(classed_data, delta, beta)

    mesh = generate_mesh(classed_data)
    mesh_cls_idxs = predict(mesh, q_weights, cache)

    if plot:
        kwargs = {"title": "Constant increase classifier"}
        plot_mesh(mesh, mesh_cls_idxs, classed_data, **kwargs)

    return num_iter


def train_q_weights(classed_data, delta, beta):
    num_dim = len(classed_data[0][0])+1
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
                    q = q + (beta/sum(x**2)) * x * omega
                    error += 1
            num_iter += 1
            if num_iter % 1000 == 0 or error == 0:
                q_weights[i] = q
                break

    return q_weights, num_iter, cache