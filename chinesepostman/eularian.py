import copy
import itertools
import random
import numpy as np
from . import dijkstra
from .my_iter import all_unique

def fleury_walk(graph, start=None, circuit=False):
    """
    Return an attempt at walking the edges of a graph.
    Tries to walk a Circuit by making random edge choices. If the route
    dead-ends, returns the route up to that point. Does not revisit
    edges.
    """
    visited = set()  # Edges

    # Begin at a random node unless start is specified:
    node = start if start else random.choice(graph.node_keys)

    route = [node]
    while len(visited) < len(graph):
        print(f"Visited {len(visited)} out of {len(graph.edges)} edges")  # Progress tracking
        # Fleury's algorithm tells us to preferentially select non-bridges
        reduced_graph = copy.deepcopy(graph)
        reduced_graph.remove_edges(visited)
        options = reduced_graph.edge_options(node)
        bridges = [k for k in options.keys() if reduced_graph.is_bridge(k)]
        non_bridges = [k for k in options.keys() if k not in bridges]
        if non_bridges:
            chosen_path = random.choice(non_bridges)
        elif bridges:
            chosen_path = random.choice(bridges)
        else:
            print(f"Dead-end reached after visiting {len(route)} nodes")  # Dead-end tracking
            break  # Reached a dead-end, no path options

        next_node = reduced_graph.edges[chosen_path].end(node)  # Other end
        visited.add(chosen_path)  # Never revisit this edge
        route.append(next_node)
        node = next_node

    print(f"Completed walk visiting {len(visited)} edges.")
    return route

def eularian_path(graph, start=None, circuit=False):
    """
    Return an Eularian Trail or Eularian Circuit through a graph, if found.
    Return the route if it visits every edge, else give up after 1000 tries.
    If `start` is set, force start at that Node.
    """
    for i in range(1, 1001):
        print(f"Attempt {i}: Starting Eulerian walk...")
        route = fleury_walk(graph, start, circuit)
        if len(route) == len(graph) + 1:  # We visited every edge
            print(f"Found Eulerian circuit in {i} attempts.")
            return route, i
        print(f"Attempt {i} failed. Current route length: {len(route)}")
    print(f"Gave up after 1000 attempts. No Eulerian circuit found.")
    return [], None  # Never found a solution

def find_dead_ends(graph):
    """
    Return a list of dead-ended edges.
    """
    single_nodes = [k for k, order in graph.node_orders.items() if order == 1]
    return set(
        [x for k in single_nodes for x in graph.edges.values() if k in (x.head, x.tail)]
    )

def build_node_pairs_knn(graph, k=2):
    """
    Builds K-Nearest Neighbor node pairs for odd-degree nodes using Dijkstra's Algorithm.
    """
    odd_nodes = graph.odd_nodes
    node_pairs = []

    # For each odd node, find its K nearest neighbors using KNN
    for node in odd_nodes:
        nearest_neighbors = knn(graph, node, k)
        for neighbor in nearest_neighbors:
            if (node, neighbor) not in node_pairs and (neighbor, node) not in node_pairs:
                node_pairs.append((node, neighbor))

    return node_pairs

def knn(graph, node, k):
    """
    K-Nearest Neighbors for a given node using Dijkstra's Algorithm.
    """
    odd_nodes = graph.odd_nodes
    distances = {}

    # Find shortest paths from the current node to all other odd nodes
    for target_node in odd_nodes:
        if target_node != node:
            cost, _ = dijkstra.find_cost((node, target_node), graph)
            distances[target_node] = cost

    sorted_neighbors = sorted(distances, key=distances.get)
    return sorted_neighbors[:k]  # Ensure only 'k' nearest neighbors are returned

def find_minimum_path_set_numpy(pair_solutions):
    """
    Use NumPy for faster operations on node pairs.
    """
    pairs = np.array([(pair[0], pair[1], cost) for pair, (cost, path) in pair_solutions.items()])
    sorted_pairs = pairs[pairs[:, 2].argsort()]

    min_cost = 0
    min_route = []
    used_nodes = set()

    for pair in sorted_pairs:
        node_a, node_b, cost = pair
        if node_a not in used_nodes and node_b not in used_nodes:
            used_nodes.update([node_a, node_b])
            min_cost += cost
            min_route.append((node_a, node_b))

        if len(used_nodes) >= len(pair_solutions):
            break

    return min_cost, min_route

def compute_pair_cost(pair, pair_solutions):
    """Helper function to compute the cost for a pair."""
    return pair_solutions[pair][0]

def build_path_sets(node_pairs, set_size):
    """Builds all possible sets of odd node pairs."""
    return (
        x for x in itertools.combinations(node_pairs, set_size) if all_unique(sum(x, ()))
    )

def unique_pairs(items):
    """Generate sets of unique pairs of odd nodes."""
    for item in items[1:]:
        pair = items[0], item
        leftovers = [a for a in items if a not in pair]
        if leftovers:
            for tail in unique_pairs(leftovers):
                yield [pair] + tail
        else:
            yield [pair]

def find_node_pair_solutions(node_pairs, graph):
    """Return path and cost for all node pairs in the path sets."""
    node_pair_solutions = {}
    for node_pair in node_pairs:
        if node_pair not in node_pair_solutions:
            cost, path = dijkstra.find_cost(node_pair, graph)
            node_pair_solutions[node_pair] = (cost, path)
            node_pair_solutions[node_pair[::-1]] = (cost, path[::-1])
    return node_pair_solutions

def build_min_set(node_solutions):
    """
    Order pairs by cheapest first and build a set by pulling pairs until every node is
    covered.
    """
    odd_nodes = set([x for pair in node_solutions.keys() for x in pair])
    sorted_solutions = sorted(node_solutions.items(), key=lambda x: x[1][0])
    path_set = []
    for node_pair, solution in sorted_solutions:
        if not all(x in odd_nodes for x in node_pair):
            continue
        path_set.append((node_pair, solution))
        for node in node_pair:
            odd_nodes.remove(node)
        if not odd_nodes:  # We've got a pair for every node
            break
    return path_set

def find_minimum_path_set(pair_sets, pair_solutions, graph):
    """Return the cheapest cost & route for all sets of node pairs, dynamically computing missing pairs."""
    cheapest_set = None
    min_cost = float('inf')
    min_route = []

    for pair_set in pair_sets:
        set_cost = 0
        temp_route = []

        for pair in pair_set:
            if pair not in pair_solutions:
                cost, path = dijkstra.find_cost(pair, graph)
                pair_solutions[pair] = (cost, path)
                pair_solutions[pair[::-1]] = (cost, path[::-1])
            set_cost += pair_solutions[pair][0]
            temp_route.append(pair_solutions[pair][1])

        if set_cost < min_cost:
            cheapest_set = pair_set
            min_cost = set_cost
            min_route = temp_route

    return cheapest_set, min_route

def add_new_edges(graph, min_route):
    """Return new graph w/ new edges extracted from minimum route."""
    new_graph = copy.deepcopy(graph)
    for node in min_route:
        for i in range(len(node) - 1):
            start, end = node[i], node[i + 1]
            cost = graph.edge_cost(start, end)  # Look up existing edge cost
            new_graph.add_edge(start, end, cost, False)  # Append new edges
    return new_graph

def make_eularian(graph, k=5):
    """Add necessary paths to the graph such that it becomes Eulerian using KNN to limit node pairs."""
    print('\tDoubling dead_ends')
    dead_ends = [x.contents for x in find_dead_ends(graph)]
    graph.add_edges(dead_ends)  # Double our dead-ends

    print(f'\tTotal dead ends found and doubled: {len(dead_ends)}')

    print('\tBuilding possible odd node pairs using KNN')
    node_pairs = build_node_pairs_knn(graph, k)
    print(f'\t\t({len(node_pairs)} pairs)')

    print('\tFinding pair solutions')
    pair_solutions = find_node_pair_solutions(node_pairs, graph)
    print(f'\t\t({len(pair_solutions)} solutions)')

    print('\tBuilding path sets')
    pair_sets = (x for x in unique_pairs(graph.odd_nodes))

    print('\tFinding cheapest route')
    cheapest_set, min_route = find_minimum_path_set(pair_sets, pair_solutions, graph)
    print('\tAdding new edges')
    return add_new_edges(graph, min_route), len(dead_ends)  # Add our new edges

if __name__ == '__main__':
    import tests.run_tests
    tests.run_tests.run(['eularian'])
