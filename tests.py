from models import Ant, Edge, Vertex, ConstructionGraph, ConstructionCycle, SolutionPath
from util import fake_random_selection


def creat_vertices():
    v1 = Vertex([1, 2], info="origin")
    v2 = Vertex([2, 1], info="destination")
    v3 = Vertex([3, 3])
    return v1, v2, v3


def test_vertex_is_correctly_attributed():
    vertices = creat_vertices()
    assert vertices[0].info == "origin"
    assert vertices[1].info == "destination"


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


def test_comparison_of_solution_paths():
    e1 = Edge(Vertex([0], info="origin"), Vertex([0.5]))
    e2 = Edge(Vertex([0.5]), Vertex([2], info="destination"))
    e3 = Edge(Vertex([0.5]), Vertex([1], info="destination"))

    sp1 = SolutionPath([e1])
    sp2 = SolutionPath([e1, e2])
    sp3 = SolutionPath([e1, e3])

    assert min(sp1, sp2) == sp2  # sp1 didn't end on a destination vertex
    assert min(sp2, sp3) == sp3  # sp3 has a shorter distance to a destination vertex


def test_conversion_of_solution_path_to_list_of_vertices():
    vertices = creat_vertices()
    sp = SolutionPath([Edge(vertices[0], vertices[2]), Edge(vertices[2], vertices[1])])

    assert sp.convert_to_list_of_vertices() == [vertices[0], vertices[2], vertices[1]]


def test_creation_of_construction_graph():
    vertices = creat_vertices()
    cg = ConstructionGraph(*vertices)

    assert cg.vertex_connections == {
        vertices[0]: {Edge(vertices[0], vertices[1]), Edge(vertices[0], vertices[2])},
        vertices[1]: {Edge(vertices[1], vertices[0]), Edge(vertices[1], vertices[2])},
        vertices[2]: {Edge(vertices[2], vertices[0]), Edge(vertices[2], vertices[1])},
    }


def test_creation_of_construction_graph_with_provided_edges():
    vertices = creat_vertices()
    edges = [Edge(vertices[0], vertices[1]), Edge(vertices[1], vertices[2])]
    cg = ConstructionGraph(*vertices, edges=edges)

    assert cg.vertex_connections == {
        vertices[0]: {Edge(vertices[0], vertices[1])},
        vertices[1]: {Edge(vertices[1], vertices[0]), Edge(vertices[1], vertices[2])},
        vertices[2]: {Edge(vertices[2], vertices[1])},
    }


def test_creation_of_construction_graph_based_on_vertex_proximity():
    vertices = [Vertex([0]), Vertex([2]), Vertex([4])]
    proximity = 2.5
    cg = ConstructionGraph(*vertices, proximity=proximity)

    assert cg.vertex_connections == {
        vertices[0]: {Edge(vertices[0], vertices[1])},
        vertices[1]: {Edge(vertices[1], vertices[0]), Edge(vertices[1], vertices[2])},
        vertices[2]: {Edge(vertices[2], vertices[1])},
    }


def test_ant_traverses_all_edges_in_traveling_salesman():
    vertices = creat_vertices()
    construction_graph = ConstructionGraph(*vertices, problem="traveling_salesman")
    ant = Ant(vertices[0])

    # inject fake random choice function to get deterministic behavior for testing
    ant.traverse_edges(construction_graph, decision_func=fake_random_selection)

    assert ant.solution_path == SolutionPath(
        [Edge(vertices[0], vertices[1]), Edge(vertices[1], vertices[2]), Edge(vertices[2], vertices[0])]
    )


def test_ant_finishes_construction_in_shortest_path():
    vertices = creat_vertices()
    construction_graph = ConstructionGraph(*vertices, problem="shortest_path")
    ant = Ant(vertices[0])

    ant.traverse_edges(construction_graph, decision_func=fake_random_selection)
    assert ant.solution_path == SolutionPath([Edge(vertices[0], vertices[1])])


def test_pheromones_placed_after_iteration():
    vertices = creat_vertices()
    cycle = ConstructionCycle(2, vertices)

    construction_graph = next(cycle)
    assert construction_graph.edges[0].pheromone > 0


def test_pheromones_placed_with_correct_evaporation():
    vertices = (
        Vertex([1], info="origin"),
        Vertex([2]),
        Vertex([3]),
    )  # 1-d vertices, distance of every round-trip should always be 4
    cycle = ConstructionCycle(2, vertices, evaporation_rate=0.1)

    construction_graph = next(cycle)  # with 0.1 being evaporated, every edge should now have 1.4 pheromone
    assert all([edge.pheromone == 1.4 for edge in construction_graph.edges])

    construction_graph = next(cycle)  # with 0.14 being evaporated, every edge should now have 0.95 pheromone
    assert all([edge.pheromone == 1.76 for edge in construction_graph.edges])


def test_pheromones_not_added_when_not_reaching_destination():
    # having a shortest-path problem, but no destination vertex defined. expect every edge to lose pheromone.
    vertices = (Vertex([1], info="origin"), Vertex([2]), Vertex([3]))
    cycle = ConstructionCycle(1, vertices, evaporation_rate=0.1, problem="shortest_path")

    construction_graph = next(cycle)
    # initial pheromone level 1.0, subtracted by the evaporation of 0.1
    assert all([edge.pheromone == 0.9 for edge in construction_graph.edges])


def test_correct_edge_probabilities_applied():
    vertices = (Vertex([1]), Vertex([2]), Vertex([3]))
    construction_graph = ConstructionGraph(*vertices)
    ant = Ant(Vertex([1]))

    feasible_edges = ant._get_feasible_edges(construction_graph)
    assert feasible_edges == {Edge(Vertex([1]), Vertex([2])), Edge(Vertex([1]), Vertex([3]))}

    edge_probabilities = ant._get_edge_probabilities(feasible_edges)
    assert edge_probabilities[Edge(Vertex([1]), Vertex([2]))] == 2 / 3
    assert edge_probabilities[Edge(Vertex([1]), Vertex([3]))] == 1 / 3


def test_saving_of_iteration_best_and_best_so_far_solutions():
    v = (Vertex([1], info="origin"), Vertex([2], info="destination"), Vertex([3]), Vertex([4]))
    cycle = ConstructionCycle(2, v)
    ants = [Ant(v[0]) for _ in range(2)]

    ants[0].solution_path = best_path = SolutionPath([Edge(v[0], v[1])])
    ants[1].solution_path = SolutionPath([Edge(v[0], v[3]), Edge(v[3], v[1])])
    cycle._get_best_solution_paths(*ants)
    assert cycle.solution_iteration_best == ants[0].solution_path
    assert cycle.solution_best_so_far == ants[0].solution_path

    ants[0].solution_path = SolutionPath([Edge(v[0], v[2]), Edge(v[2], v[1])])
    ants[1].solution_path = SolutionPath([Edge(v[0], v[3]), Edge(v[3], v[1])])
    cycle._get_best_solution_paths(*ants)
    assert cycle.solution_iteration_best == ants[0].solution_path
    assert cycle.solution_best_so_far == best_path
