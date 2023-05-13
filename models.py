from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

# TODO:
# 1. implement ant and edges objects and their interaction                          (done)
# 2. make ants travel over all vertices once (randomly)                             (done)
# 3. place pheromones after each solution construction cycle (with evaporation)
# 4. implement a decision function based on edge quality and probabilities


@dataclass(frozen=True)
class Vertex:
    coord: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return False
        return (self.coord == other.coord).all()

    def __hash__(self):
        return hash(str(self.coord))


class Ant:
    def __init__(self, position: Vertex):
        self.origin_position = position
        self.position = position
        self.traveled_edges = []  # type: List[Edge]

    def travel_all_edges_randomly(self, cg: ConstructionGraph, *, random_seed: int = None):
        rng = np.random.default_rng(seed=random_seed)

        while len(self.traveled_edges) < len(cg.vertex_connections):
            edges = cg.vertex_connections[self.position]
            feasible_edges = set(filter(lambda e: e not in self.traveled_edges, edges))
            if (len(feasible_edges) > 1) and (self.origin_position != self.position):
                feasible_edges - set(filter(lambda e: self.origin_position in e, feasible_edges))
            edge_choice = rng.choice(list(feasible_edges), 1)[0]

            self.traveled_edges.append(edge_choice)
            self.position = edge_choice.get_destination_vertex(self.position)


class Edge:
    def __init__(self, i: Vertex, j: Vertex):
        self.i = i
        self.j = j
        self._pheromone = 0.  # type: float

    def __eq__(self, other):
        """Edges are non-directional, meaning Edge(i, j) == Edge(j, i)."""
        if not isinstance(other, Edge):
            return False
        return ((self.i == other.i) and (self.j == other.j)) or (
                (self.i == other.j) and (self.j == other.i))

    def __hash__(self):
        """Sorting vertices in order to assure same hash independent of Vertex order"""
        return hash(str(sorted([str(self.i), str(self.j)])))

    def __repr__(self):
        return f"Edge: ({str(self.i)}, {str(self.j)})"

    def get_destination_vertex(self, curr_vertex: Vertex):
        if curr_vertex == self.i:
            return self.j
        if curr_vertex == self.j:
            return self.i

    def add_pheromone_value(self, value: float):
        self._pheromone += value


class ConstructionGraph:
    """A dict object with keys representing vertices, and values representing all possible edges.

    Mutable Edges are supposed to be accessed and altered through this object.
    """

    def __init__(self, *vertices: Vertex):
        self.vertex_connections = self.get_edges_from_all_vertices_connections(*vertices)  # type: dict

    @staticmethod
    def get_edges_from_all_vertices_connections(*vertices: Vertex):
        construction_dict = {v: set(map(lambda other: Edge(v, other), filter(lambda other: other != v, vertices)))
                             for v in vertices}
        return construction_dict
