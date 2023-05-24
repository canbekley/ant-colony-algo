from typing import List
import numpy as np

from models import Vertex, ConstructionGraph


def random_selection(arr, *args, **kwargs):
    """Returns a random element of a sorted array."""
    return np.random.choice(sorted(list(arr)), 1)[0]


def fake_random_selection(arr, *args, **kwargs):
    """Always returns the first element of a sorted array."""
    return sorted(list(arr))[0]


def construct_spherical_graph(edge_proximity: float = 1.5):
    mesh = np.mgrid[0:10, 0:10]
    array_space = list(zip(mesh[0].flatten(), mesh[1].flatten()))
    sphere_center = (5, 5)
    sphere_radius = 3
    array_space_in_sphere = list(
        filter(
            lambda arr: ((arr[0] - sphere_center[0]) ** 2 + (arr[1] - sphere_center[1]) ** 2) <= sphere_radius**2,
            array_space,
        )
    )
    bottom = np.amin(array_space_in_sphere, axis=0)[1]
    top = np.amax(array_space_in_sphere, axis=0)[1]
    vertices = list(
        map(
            lambda arr: Vertex(arr, info="origin" if arr[1] == bottom else "destination" if arr[1] == top else None),
            array_space_in_sphere,
        )
    )
    edges = ConstructionGraph.get_all_combinatorial_edges(*vertices, proximity=edge_proximity)
    return vertices, edges


def generate_random_2dimensional_vertices_and_edges(
    num_vertices: int = 10, edge_proximity: float = 3, random_seed: int | None = None
):
    np.random.seed(random_seed)

    mesh = np.mgrid[1:7, 1:7]
    array_space = list(zip(mesh[0].flatten(), mesh[1].flatten()))
    vertices = list(
        map(
            lambda x: Vertex(x[1], info="origin" if x[0] == 0 else "destination" if x[0] == 1 else None),
            enumerate(np.array(array_space)[np.random.choice(len(array_space), num_vertices, replace=False)]),
        )
    )
    edges = ConstructionGraph.get_all_combinatorial_edges(*vertices, proximity=edge_proximity)
    return vertices, edges
