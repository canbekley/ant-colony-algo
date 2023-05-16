from __future__ import annotations

from collections.abc import Iterable, Sized
from functools import reduce
from typing import List, Tuple, Callable, Union
import numpy as np

from util import random_choice


# TODO:
# 1. implement ant and edges objects and their interaction                          (done)
# 2. make ants travel over all vertices once (randomly)                             (done)
# 3. place pheromones after each solution construction cycle (with evaporation)     (done)
# 4. implement a decision function based on edge quality and probabilities


class Ant:
    def __init__(self, position: Vertex):
        self.origin_position = position
        self.position = position
        self.traveled_edges = []  # type: List[Edge]

    @property
    def traveled_distance(self) -> float:
        return reduce(lambda a, b: a + b, map(lambda edge: edge.length, self.traveled_edges))

    def travel_all_edges_randomly(self, cg: ConstructionGraph, *, random_choice_func: Callable = random_choice):
        while len(self.traveled_edges) < len(cg.vertex_connections):
            edges = cg.vertex_connections[self.position]
            feasible_edges = set(filter(lambda e: e not in self.traveled_edges, edges))
            if (len(feasible_edges) > 1) and (self.origin_position != self.position):
                feasible_edges -= set(filter(lambda e: self.origin_position in e, feasible_edges))
            edge_choice = random_choice_func(feasible_edges)

            self.traveled_edges.append(edge_choice)
            self.position = edge_choice.get_destination_vertex(self.position)

    def get_pheromone_delta(self, edge: Edge, **params: float) -> float:
        pheromone_delta_constant = params['pheromone_delta_constant']
        pheromone_delta = 0. if not self._edge_passed(edge) else pheromone_delta_constant / self.traveled_distance
        return pheromone_delta

    def _edge_passed(self, edge: Edge) -> bool:
        return edge in self.traveled_edges


class Vertex(tuple):
    def __sub__(self, other: Union[Iterable, Sized]) -> Vertex:
        """Generates a new Vertex from array subtraction."""
        if not isinstance(other, (Iterable, Sized)):
            raise TypeError('Subtracted object is not an iterable or is not sized.')
        if len(other) != len(self):
            raise ValueError('Different sized arrays cannot be subtracted.')
        return Vertex(m - n for m, n in zip(self, other))


class Edge:
    def __init__(self, i: Vertex, j: Vertex):
        self.i = i
        self.j = j
        self.pheromone = 0.  # type: float

    def __contains__(self, item) -> bool:
        return item in [self.i, self.j]

    def __eq__(self, other) -> bool:
        """Edges are non-directional, meaning Edge(i, j) == Edge(j, i)."""
        if not isinstance(other, Edge):
            return False
        return ((self.i == other.i) and (self.j == other.j)) or (
                (self.i == other.j) and (self.j == other.i))

    def __lt__(self, other) -> bool:
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
    def length(self):
        return np.linalg.norm(self.i - self.j)

    def get_destination_vertex(self, curr_vertex: Vertex):
        if curr_vertex == self.i:
            return self.j
        if curr_vertex == self.j:
            return self.i

    def calculate_new_pheromone_levels(self, pheromone_delta_sum: float, **params: float):
        self.pheromone = (1 - params['evaporation_rate']) * self.pheromone + pheromone_delta_sum


class ConstructionGraph:
    """Contains a dict object with keys representing vertices, and values representing all possible edges.

    Mutable Edges are supposed to be accessed and altered through this object.
    """

    def __init__(self, *vertices: Vertex):
        self.vertices = vertices  # type: Tuple[Vertex]
        self.edges = self.get_all_edges()  # type: List[Edge]
        self.vertex_connections = self.get_all_vertex_connections()  # type: dict

    def get_all_edges(self) -> List[Edge]:
        edges = map(lambda v1: list(map(lambda v2: Edge(v1, v2), self.vertices[self.vertices.index(v1) + 1:])),
                    self.vertices)
        edges_reduced = reduce(lambda a, b: a + b, edges)
        return edges_reduced

    def get_all_vertex_connections(self) -> dict:
        vertex_connections = {vertex: set(filter(lambda edge: vertex in edge, self.edges)) for vertex in self.vertices}
        return vertex_connections

    def lay_pheromone(self, *ants: Ant, **params: float):
        for edge in self.edges:
            pheromone_deltas = map(lambda ant: ant.get_pheromone_delta(edge, **params), ants)
            pheromone_delta_sum = sum(list(pheromone_deltas))
            edge.calculate_new_pheromone_levels(pheromone_delta_sum, **params)


class ConstructionCycle:

    def __init__(self,
                 num_ants: int,
                 ant_position: Vertex,
                 *vertices: Vertex,
                 evaporation_rate: float = 0.1,
                 pheromone_delta_constant: float = 1.,
                 control_param_pheromone: float = 1.,
                 control_param_distance: float = 1.):
        self.num_ants = num_ants
        self.ant_position = ant_position
        self.params = {'evaporation_rate': evaporation_rate,
                       'pheromone_delta_constant': pheromone_delta_constant,
                       'control_param_pheromone': control_param_pheromone,
                       'control_param_distance': control_param_distance}

        self.construction_graph = ConstructionGraph(*vertices)

    def __next__(self) -> ConstructionGraph:
        ants = [Ant(self.ant_position) for _ in range(self.num_ants)]
        list(map(lambda ant: ant.travel_all_edges_randomly(self.construction_graph), ants))
        self.construction_graph.lay_pheromone(*ants, **self.params)
        return self.construction_graph

    def __iter__(self) -> ConstructionCycle:
        return self
