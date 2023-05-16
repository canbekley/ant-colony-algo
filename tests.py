from models import Ant, Edge, Vertex, ConstructionGraph, ConstructionCycle
from util import fake_random_selection


def creat_vertices():
    v1 = Vertex([1, 2])
    v2 = Vertex([2, 1])
    v3 = Vertex([3, 3])
    return v1, v2, v3


def test_edge_is_non_directional():
    vertices = creat_vertices()

    assert Edge(vertices[0], vertices[1]) == Edge(vertices[1], vertices[0])


def test_edge_is_correctly_hashed():
    vertices = creat_vertices()
    edge1 = Edge(vertices[0], vertices[1])
    edge2 = Edge(vertices[1], vertices[0])  # same logical edge
    hashed_dict = dict()
    hashed_dict[edge1] = 1
    hashed_dict[edge2] = 5

    assert hashed_dict == {edge1: 5}
    assert hashed_dict == {edge2: 5}


def test_edge_has_length():
    edge = Edge(Vertex([1, 1]), Vertex([1, 2]))
    assert edge.length == 1


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

    # inject fake random choice function to get deterministic behavior for testing
    ant.travel_edges(construction_graph, decision_func=fake_random_selection)

    assert ant.traveled_edges == [Edge(vertices[0], vertices[1]),
                                  Edge(vertices[1], vertices[2]),
                                  Edge(vertices[2], vertices[0])]


def test_pheromones_placed_after_iteration():
    vertices = creat_vertices()
    cycle = ConstructionCycle(2, vertices[0], *vertices)

    construction_graph = next(cycle)
    assert construction_graph.edges[0].pheromone > 0


def test_pheromones_placed_with_correct_evaporation():
    vertices = [Vertex([1]), Vertex([2]), Vertex([3])]  # 1-d vertices, distance of every round-trip should always be 4
    cycle = ConstructionCycle(2, vertices[0], *vertices)  # 2 ants have a combined pheromone delta of 0.5 ( 1/4 + 1/4 )

    construction_graph = next(cycle)  # starting with 0. pheromone, every edge should now have 0.5 pheromone
    assert all([edge.pheromone == 0.5 for edge in construction_graph.edges])

    construction_graph = next(cycle)  # 0.05 being evaporated, every edge should now have 0.95 pheromone
    assert all([edge.pheromone == 0.95 for edge in construction_graph.edges])


def test_correct_edge_probabilities_applied():
    vertices = [Vertex([1]), Vertex([2]), Vertex([3])]
    construction_graph = ConstructionGraph(*vertices)
    ant = Ant(Vertex([1]))

    feasible_edges = ant._get_feasible_edges(construction_graph)
    assert feasible_edges == {Edge(Vertex([1]), Vertex([2])), Edge(Vertex([1]), Vertex([3]))}

    for edge in feasible_edges:
        edge._pheromone = 1.
    edge_probabilities = ant._get_edge_probabilities(feasible_edges)
    assert edge_probabilities[Edge(Vertex([1]), Vertex([2]))] == 2/3
    assert edge_probabilities[Edge(Vertex([1]), Vertex([3]))] == 1/3
