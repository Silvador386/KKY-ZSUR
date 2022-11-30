import numpy as np
from plot import plot_mesh, generate_mesh
from utils import L2_distance_matrix


def vector_quantization(classed_data, center_data, plot=True):
    classed_data = np.copy(classed_data)
    mesh = generate_mesh(classed_data)
    dist_matrix = L2_distance_matrix(mesh, center_data)

    mesh_cls_rule = np.argmin(dist_matrix, axis=1)

    if plot:
        kwargs = {"title": "Vector quantization classifier"}
        plot_mesh(mesh, mesh_cls_rule, classed_data, **kwargs)

