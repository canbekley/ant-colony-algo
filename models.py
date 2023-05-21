from __future__ import annotations

from collections.abc import Iterable, Sized
from functools import reduce
from typing import List, Tuple, Set, Callable, Union, Sequence, Any, Dict
import numpy as np


class SolutionPath(list):
    """A partial or complete solution construction, described though a sequence of Edges."""
    def __lt__(self, other):
        if not isinstance(other, SolutionPath):
            return True
        return self.distance() < other.distance()

    def append(self, __object: Edge) -> None:
        super().append(__object)

    def distance(self) -> float:
        if not len(self):
            return 0.
        return reduce(lambda a, b: a + b, map(lambda edge: edge.length, self))


class Ant:
    """An agent able to construct solution paths in a solution space."""
    def __init__(self, position: Vertex):
        self.origin_position = position
        self.position = position
        self.solution_path = SolutionPath()

    def traverse_edges(self, cg: ConstructionGraph, *, decision_func: Callable = None, random_seed: int = None):
        np.random.seed(random_seed)  # for deterministic testing
        decision_func = self._probabilistic_pathing if decision_func is None else decision_func

        while True:
            feasible_edges = self._get_feasible_edges(cg)
            edge_choice = decision_func(feasible_edges)
            self.solution_path.append(edge_choice)
            self.position = edge_choice.get_destination_vertex(self.position)
            if self._finish_construction(cg):
                break

    def get_pheromone_delta(self, edge: Edge, delta_constant: float) -> float:
        return 0. if not self._edge_passed(edge) else delta_constant / self.solution_path.distance()

    def _get_feasible_edges(self, cg: ConstructionGraph) -> Set[Edge]:
        edges = cg.vertex_connections[self.position]
        feasible_edges = set(filter(lambda e: e not in self.solution_path, edges))
        if (len(feasible_edges) > 1) and (self.origin_position != self.position):
            feasible_edges -= set(filter(lambda e: self.origin_position in e, feasible_edges))

        return feasible_edges

    def _probabilistic_pathing(self, feasible_edges: Set[Edge]) -> Edge:
        edge_probabilities = self._get_edge_probabilities(feasible_edges)
        edge_choice = np.random.choice(list(edge_probabilities.keys()), 1, p=list(edge_probabilities.values()))[0]
        return edge_choice

    def _finish_construction(self, cg: ConstructionGraph) -> bool:
        if cg.problem == 'traveling_salesman':
            return len(self.solution_path) >= len(cg.vertex_connections)
        if cg.problem == 'shortest_path':
            return self.position.is_path_destination or (len(self.solution_path) >= len(cg.vertex_connections))

    def _edge_passed(self, edge: Edge) -> bool:
        return edge in self.solution_path

    @staticmethod
    def _get_edge_probabilities(feasible_edges: Set[Edge]) -> Dict[Edge, float]:
        edge_qualities = {edge: edge.edge_quality for edge in list(feasible_edges)}
        edge_quality_sum = sum(edge_qualities.values())
        edge_probabilities = {edge: edge_score / edge_quality_sum if edge_quality_sum else 1 / len(feasible_edges)
                              for edge, edge_score in edge_qualities.items()}
        return edge_probabilities


class Vertex(tuple):
    """Immutable n-dimensional waypoints, representing traversable points in the solution construction."""
    def __new__(cls, *args, **kwargs):
        return tuple.__new__(Vertex, *args, **kwargs)

    def __init__(self, *args, is_path_destination: bool = False):
        super().__init__()
        self._is_path_destination = is_path_destination

    def __sub__(self, other: Union[Iterable, Sized]) -> Vertex:
        """Generates a new Vertex from array subtraction."""
        if not isinstance(other, (Iterable, Sized)):
            raise TypeError('Subtracted object is not an iterable or is not sized.')
        if len(other) != len(self):
            raise ValueError('Different sized arrays cannot be subtracted.')
        return Vertex([m - n for m, n in zip(self, other)], is_path_destination=self.is_path_destination)

    @property
    def is_path_destination(self):
        return self._is_path_destination


class Edge:
    """The connection between the two vertices i and j.

    Contains the pheromone variable.
    """
    def __init__(self,
                 i: Vertex,
                 j: Vertex,
                 control_param_pheromone: float = 1.,
                 control_param_distance: float = 1.):
        self.i = i
        self.j = j
        self._control_param_pheromone = control_param_pheromone
        self._control_param_distance = control_param_distance
        self._pheromone = 0.  # type: float

    def __contains__(self, item) -> bool:
        return item in [self.i, self.j]

    def __eq__(self, other: Any) -> bool:
        """Edges are non-directional, meaning Edge(i, j) == Edge(j, i)."""
        if not isinstance(other, Edge):
            return False
        return ((self.i == other.i) and (self.j == other.j)) or (
                (self.i == other.j) and (self.j == other.i))

    def __lt__(self, other: Any) -> bool:
        """A generic way to make Edges sortable."""
        if not isinstance(other, Edge):
            return True
        return (sum(self.i) + sum(self.j)) < (sum(other.i) + sum(other.j))

    def __hash__(self) -> int:
        """Sorting vertices here to assure same resulting hashes, independent of Vertex order."""
        return hash(str(sorted([str(self.i), str(self.j)])))

    def __repr__(self) -> str:
        return f"Edge({str(self.i)}, {str(self.j)})"

    @property
    def pheromone(self):
        return self._pheromone

    @property
    def length(self):
        return np.linalg.norm(self.i - self.j)

    @property
    def distance_score(self):
        """Distance score η(i, j)."""
        return 1 / self.length

    @property
    def edge_quality(self):
        return (self.pheromone ** self._control_param_pheromone) * (self.distance_score ** self._control_param_distance)

    def get_destination_vertex(self, curr_vertex: Vertex):
        if curr_vertex == self.i:
            return self.j
        if curr_vertex == self.j:
            return self.i

    def calculate_new_pheromone_levels(self, pheromone_delta_sum: float, evaporation_rate: float):
        self._pheromone = (1 - evaporation_rate) * self.pheromone + pheromone_delta_sum


class ConstructionGraph:
    """Contains a dict object with keys representing vertices, and values representing all possible edges.

    Mutable Edges are supposed to be accessed and altered through this object.
    """
    def __init__(self, *vertices: Vertex, edges: List[Edge] | None = None, problem: str = 'traveling_salesman'):
        self.vertices = vertices  # type: Tuple[Vertex]
        self.edges = edges if edges else self.get_all_combinatorial_edges()  # type: List[Edge]
        self.vertex_connections = self.get_all_vertex_connections()  # type: dict
        self.problem = problem

    def get_all_combinatorial_edges(self) -> List[Edge]:
        edges = map(lambda v1: list(map(lambda v2: Edge(v1, v2), self.vertices[self.vertices.index(v1) + 1:])),
                    self.vertices)
        edges_reduced = reduce(lambda a, b: a + b, edges)
        return edges_reduced

    def get_all_vertex_connections(self) -> dict:
        vertex_connections = {vertex: set(filter(lambda edge: vertex in edge, self.edges)) for vertex in self.vertices}
        return vertex_connections


class ConstructionCycle:
    """Initializer for Ant-System algorithm.

    Generates a new optimization cycle on each next() call.
    """
    def __init__(self,
                 num_ants: int,
                 ant_position: Vertex,
                 vertices: Tuple[Vertex, ...],
                 edges: List[Edge] | None = None,
                 *,
                 evaporation_rate: float = 0.1,
                 delta_constant: float = 1.,
                 control_param_pheromone: float = 1.,
                 control_param_distance: float = 1.,
                 problem: str = 'traveling_salesman'):
        self.num_ants = num_ants
        self.ant_position = ant_position
        self.params = {'evaporation_rate': evaporation_rate,
                       'delta_constant': delta_constant,
                       'control_param_pheromone': control_param_pheromone,
                       'control_param_distance': control_param_distance,
                       'problem': problem}
        self.construction_graph = ConstructionGraph(*vertices, edges=edges, problem=self.params['problem'])
        self.solution_best_so_far = None  # type: SolutionPath | None
        self.solution_iteration_best = None  # type: SolutionPath | None

    def __next__(self) -> ConstructionGraph:
        ants = [Ant(self.ant_position) for _ in range(self.num_ants)]
        self._construct_solution(*ants)
        self._update_pheromones(*ants)
        return self.construction_graph

    def __iter__(self) -> ConstructionCycle:
        return self

    def _construct_solution(self, *ants: Ant):
        for ant in ants:
            ant.traverse_edges(self.construction_graph)
        self._get_best_solution_paths(*ants)

    def _get_best_solution_paths(self, *ants: Ant):
        self.solution_iteration_best = min(ant.solution_path for ant in ants)
        self.solution_best_so_far = min(self.solution_best_so_far, self.solution_iteration_best)

    def _update_pheromones(self, *ants: Ant):
        for edge in self.construction_graph.edges:
            pheromone_deltas = map(lambda ant: ant.get_pheromone_delta(edge, self.params['delta_constant']), ants)
            pheromone_delta_sum = sum(list(pheromone_deltas))
            edge.calculate_new_pheromone_levels(pheromone_delta_sum, self.params['evaporation_rate'])
