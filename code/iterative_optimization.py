import numpy as np
from utils import L2_distance_matrix


def optimize(clustered_data, center_data):
    num_cls = len(clustered_data)
    error_f = np.zeros(center_data.shape[0])
    for i, (cls_data, cls_center) in enumerate(zip(clustered_data, center_data)):
        dists_matrix = L2_distance_matrix(cls_data, cls_center[:, np.newaxis].T)
        error_f[i] = np.sum(dists_matrix)
    total_error = np.sum(error_f)

    for i, (cls_data, cls_center) in enumerate(zip(clustered_data, center_data)):
        for dp_idx, data_point in enumerate(cls_data):
            if len(cls_data) == 1:
                continue
            A_i = len(cls_data)/(len(cls_data)-1) * np.sum(data_point**2 - cls_center**2)
            A_js_with_idx = []
            for j, (cls_data2, cls_center2) in enumerate(zip(clustered_data, center_data)):
                if i == j:
                    continue
                A_j = len(cls_data2) / (len(cls_data2) + 1) * np.sum(data_point ** 2 - cls_center2 ** 2)
                A_js_with_idx.append([A_j, j])

            A_min = A_js_with_idx[np.argmin(np.array(A_js_with_idx)[:, ])]
            if A_i > A_min[0]:
                clustered_data[A_min[1]] = np.vstack([clustered_data[A_min[1]], data_point])
                cls_data = np.delete(cls_data, dp_idx)


                cls_center = np.average(cls_data)
                center_data[A_min[1]] = np.average(clustered_data[A_min[1]])

                error_f2 = np.zeros(center_data.shape[0])
                for i3, (cls_data3, cls_center3) in enumerate(zip(clustered_data, center_data)):
                    dists_matrix = L2_distance_matrix(cls_data3, cls_center3[:, np.newaxis].T)
                    error_f2[i] = np.sum(dists_matrix)


    return clustered_data

