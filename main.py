import argparse
import sys
import numpy as np
import pandas as pd  # Import pandas for writing to Excel
import data.data
from chinesepostman import eularian, network


def setup_args():
    """Setup argparse to take graph name argument."""
    parser = argparse.ArgumentParser(description='Find an Eulerian Circuit.')
    parser.add_argument('graph', nargs='?', help='Name of graph to load')
    parser.add_argument(
        'start',
        nargs='?',
        type=int,
        help='The starting node. Random if none provided.'
    )
    args = parser.parse_args()
    return args


def convert_graph_to_sets_and_matrix(graph, num_nodes):
    directed_edges = []
    undirected_edges = []

    # Create an empty distance matrix with 0s
    distance_matrix = np.zeros((num_nodes, num_nodes))

    # Iterate over the edges in the graph
    for edge in graph.edges.values():
        u, v, weight, directed = edge.head, edge.tail, edge.weight, edge.directed
        if directed:
            directed_edges.append((u, v, weight))  # Directed edge (u, v, weight)
            distance_matrix[u - 1][v - 1] = weight  # Directed distance
        else:
            if (v, u) not in undirected_edges:
                undirected_edges.append((u, v, weight))  # Undirected edge (u, v, weight)
                undirected_edges.append((v, u, weight))  # Add the reverse with weight
                distance_matrix[u - 1][v - 1] = weight
                distance_matrix[v - 1][u - 1] = weight  # Symmetric for undirected

    return directed_edges, undirected_edges, distance_matrix


def save_to_excel(directed_edges, undirected_edges, distance_matrix, filename='graph_output.xlsx'):
    """Save directed set, undirected set, and distance matrix to an Excel file."""
    with pd.ExcelWriter(filename) as writer:
        # Convert directed edges to a DataFrame (with weight) and save to sheet
        df_directed = pd.DataFrame(directed_edges, columns=['From', 'To', 'Weight'])
        df_directed.to_excel(writer, sheet_name='Directed_Edges', index=False)

        # Convert undirected edges to a DataFrame (with weight) and save to sheet
        undirected_edges_set = set(undirected_edges)  # Remove duplicates
        df_undirected = pd.DataFrame(sorted(undirected_edges_set), columns=['From', 'To', 'Weight'])
        df_undirected.to_excel(writer, sheet_name='Undirected_Edges', index=False)

        # Save distance matrix to a sheet with rows and columns starting from 1
        df_matrix = pd.DataFrame(distance_matrix,
                                 index=range(1, distance_matrix.shape[0] + 1),  # Set index starting from 1
                                 columns=range(1, distance_matrix.shape[1] + 1))  # Set columns starting from 1
        df_matrix.to_excel(writer, sheet_name='Distance_Matrix', index=True)

    print(f'Data saved to {filename}')


def main():
    """Main function to compute and return the Eulerized graph and solve the Chinese Postman Problem."""
    edges = None
    args = setup_args()
    graph_name = args.graph
    try:
        print(f'Loading graph: {graph_name}')
        edges = getattr(data.data, graph_name)
    except (AttributeError, TypeError):
        available = [x for x in dir(data.data) if not x.startswith('__')]
        print(
            '\nInvalid graph name.'
            ' Available graphs:\n\t{}\n'.format('\n\t'.join(available))
        )
        sys.exit()

    original_graph = network.Graph(edges)

    print(f'<{len(original_graph.edges)}> edges loaded')
    if not original_graph.is_eularian:
        print('Converting to Eulerian path...')
        graph, num_dead_ends = eularian.make_eularian(original_graph)
        print('Conversion complete')
        print(f'\tAdded {len(graph.edges) - len(original_graph.edges) + num_dead_ends} edges')
        print(f'\tTotal cost is {graph.total_cost}')
    else:
        graph = original_graph
        print("Graph is already Eulerian")

    # Number of nodes (intersections)
    num_nodes = len(original_graph.nodes)

    # Convert the graph to sets of directed and undirected edges and a distance matrix
    directed_edges, undirected_edges, distance_matrix = convert_graph_to_sets_and_matrix(graph, num_nodes)

    # Save the directed, undirected edges and distance matrix to an Excel file
    save_to_excel(directed_edges, undirected_edges, distance_matrix, 'graph_output.xlsx')

    # Attempt to solve Eulerian Circuit
    print('Attempting to solve Eulerian Circuit...')
    route, attempts = eularian.eularian_path(graph, args.start)
    if not route:
        print(f'\tGave up after <{attempts}> attempts.')
    else:
        print(f'\tSolved in <{attempts}> attempts')
        print(f'Solution: (<{len(route) - 1}> edges)')
        print(f'\t{route}')

    return graph  # This returns the modified Eulerized graph


if __name__ == '__main__':
    main()
