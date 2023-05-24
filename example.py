import argparse
from pprint import pprint

import util
from plot import plot_construction_graph
from models import ConstructionCycle


def main(n_ants: int, n_cycles: int, n_vertices: int = None, seed: int = None, problem: str = "shortest_path", /):
    vertices, edges = util.generate_random_2dimensional_vertices_and_edges(n_vertices, random_seed=seed)
    print(
        f"Initializing with {len(vertices)} vertices and {len(edges)} edges.\n"
        f"Starting point: {[v for v in vertices if v.is_origin][0]}, "
        f"Destination {[v for v in vertices if v.is_destination][0]}\n"
    )

    cc = ConstructionCycle(n_ants, vertices, edges, problem=problem)
    for i in range(n_cycles):
        next(cc)
        pprint(f"iteration best: {cc.solution_iteration_best}")
        pprint(f"best so far: {cc.solution_best_so_far}")
        pprint(f"Pheromone values: {cc.construction_graph.edges}")
        plot_construction_graph("graph_step{i:02d}.png", cc, colored_pheromones=True)
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--ants", type=int, help="number of ants per optimization cycle.")
    parser.add_argument("-c", "--cycles", type=int, help="number of optimization cycles.")
    parser.add_argument("-v", "--vertices", type=int, help="number of vertices to generate.")
    parser.add_argument("-s", "--seed", type=int, help="a random seed to make the program deterministic.")
    parser.add_argument("-pr", "--problem", type=str, help='one of: "traveling_salesman" or "shortest_path".')

    args = parser.parse_args()
    main(args.ants, args.cycles, args.vertices, args.seed, args.problem)
