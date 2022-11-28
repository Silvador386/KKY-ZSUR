import random
import numpy as np
from utils import L2_distance_matrix, find_cls_data_centers
from plot import plot_2D


def k_means_div(data, num_cls, plot=False):
    data = data.copy()
    center_idxs_list = [[value] for value in random.sample(range(max(data.shape[0], num_cls)), num_cls)]
    center_data = data[center_idxs_list].reshape(-1, 2)

    while True:
        dists_matrix = L2_distance_matrix(data, center_data)

        for idx, row in enumerate(dists_matrix):
            min = np.argmin(row)
            if idx not in center_idxs_list[min]:
                center_idxs_list[min].append(idx)

        classed_data = []
        for center_idxs in center_idxs_list:
            classed_data.append(data[center_idxs])

        new_center_data = find_cls_data_centers(classed_data)

        # new_center_data = []
        # for center_idxs in center_idxs_list:
        #     mid_vector = data[center_idxs, :]
        #     avg = np.average(mid_vector, axis=0)
        #     new_center_data.append(avg)

        new_center_data = np.array(new_center_data)

        if (new_center_data == center_data).all():
            break

        center_idxs_list = [[] for _ in range(num_cls)]
        center_data = new_center_data


    if plot:
        data2plot_named = {f"Center: {center[0]:.2f}, {center[1]:.2f}": data
                           for data, center in zip(classed_data, new_center_data)}
        data2plot_named["title"] = "K - Means"
        plot_2D(**data2plot_named)

    return classed_data, new_center_data



