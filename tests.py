import numpy as np

from models import Ant, Edge, Vertex, ConstructionGraph


def creat_vertices():
    v1 = Vertex(np.array([1, 2]))
    v2 = Vertex(np.array([2, 1]))
    v3 = Vertex(np.array([3, 3]))
    return v1, v2, v3


def test_edge_is_non_directional():
    vertices = creat_vertices()

    assert Edge(vertices[0], vertices[1]) == Edge(vertices[1], vertices[0])


def test_successful_creation_of_construction_graph():
    vertices = creat_vertices()
    cg = ConstructionGraph(*vertices)

    assert cg.vertex_connections == {vertices[0]: {Edge(vertices[0], vertices[1]), Edge(vertices[0], vertices[2])},
                                     vertices[1]: {Edge(vertices[1], vertices[0]), Edge(vertices[1], vertices[2])},
                                     vertices[2]: {Edge(vertices[2], vertices[0]), Edge(vertices[2], vertices[1])}}


def test_ant_travels_through_all_edges():
    vertices = creat_vertices()
    construction_graph = ConstructionGraph(*vertices)
    ant = Ant(vertices[0])

    ant.travel_all_edges_randomly(construction_graph, random_seed=42)

    assert ant.traveled_edges == [Edge(vertices[0], vertices[2]),
                                  Edge(vertices[2], vertices[1]),
                                  Edge(vertices[1], vertices[0])]
