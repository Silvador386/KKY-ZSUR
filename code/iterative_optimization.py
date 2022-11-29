import numpy as np
from utils import L2_distance_matrix
from plot import plot_2D


def optimize(classed_data, center_data, plot=False):
    data = np.array([np.append(row, cls_idx) for cls_idx, data_class in enumerate(classed_data) for row in data_class])
    center_data = np.array(center_data)
    num_R = len(center_data)
    cls_sizes = np.array([cls_data.shape[0] for cls_data in classed_data])

    cls_error = np.zeros(center_data.shape[0])
    for i, (cls_data, cls_center) in enumerate(zip(classed_data, center_data)):
        dists_matrix = L2_distance_matrix(cls_data, cls_center[:, np.newaxis].T)
        cls_error[i] = np.sum(dists_matrix)

    previous_error = -1
    while True:
        total_error = np.sum(cls_error)
        if total_error == previous_error:
            break
        previous_error = total_error

        for dp_idx, data_point in enumerate(data):
            dp_cls = int(data_point[2])

            if cls_sizes[dp_cls] == 1:
                continue

            dists = L2_distance_matrix(data_point[:2, np.newaxis].T, center_data)
            change_v = [-1 if i == dp_cls else 1 for i in range(num_R)]
            size_scale = cls_sizes / np.array(cls_sizes + change_v)

            A = (size_scale * dists).reshape(-1)
            A_i_val = A[dp_cls]
            A_j_val = np.min(A[A != A_i_val])
            A_j_cls = int(np.where(A == A_j_val)[0])

            if A_i_val > A_j_val:
                data[dp_idx][-1] = A_j_cls
                cls_error[dp_cls] -= A_i_val
                cls_error[A_j_cls] += A_j_val

                cls_sizes[dp_cls] -= 1
                cls_sizes[A_j_cls] += 1

                center_data[dp_cls] = np.average(data[np.where(data[:, -1] == dp_cls)][:, :-1], axis=0)
                center_data[A_j_cls] = np.average(data[np.where(data[:, -1] == A_j_cls)][:, :-1], axis=0)

    data = data[data[:, -1].argsort()]
    classed_data = np.split(data[:, :-1], np.where(np.diff(data[:, -1]))[0]+1)

    if plot:
        data2plot_named = {f"Center: {center[0]:.2f}, {center[1]:.2f}": data
                           for data, center in zip(classed_data, center_data)}
        data2plot_named["title"] = "Iterative Optimization"
        plot_2D(**data2plot_named)

    return classed_data, center_data, total_error

