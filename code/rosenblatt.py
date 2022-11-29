import numpy as np
from utils import L2_distance_matrix
from plot import generate_mesh, plot_mesh


def rosenblatt(classed_data, plot=True):
    delta = 0
    lr_constant = 1
    num_dim = len(classed_data[0][0])
    classed_data = np.copy(classed_data)
    cls_labels = [i for i, single_data in enumerate(classed_data) for _ in single_data]
    merged_data = np.concatenate(classed_data)

    q_weights = [np.random.randint(-10, 10, size=num_dim+1) for _ in classed_data]

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
            if num_iter % 500 == 0 or error == 0:
                q_weights[i] = q
                break

    mesh = generate_mesh(classed_data)
    mesh_cls_idxs = np.zeros(mesh.shape[0])-1
    for i, data_point in enumerate(mesh):
        x = np.array([1, *data_point])
        inequalities = np.array([q.T @ x >= 0 for q in q_weights])
        if sum(inequalities) == 1:
            mesh_cls_idxs[i] = np.argwhere(inequalities)

    if plot:
        kwargs = {"title": "Rosenblatt classifier"}
        plot_mesh(mesh, mesh_cls_idxs, classed_data, **kwargs)




