TEST = """
import networkx as nx
import timeit
from itertools import combinations

def setup_graph():
    # Create an Erdős-Rényi graph with 100 nodes and a probability of 0.1 for edge creation
    G = nx.erdos_renyi_graph(100, 0.1)
    return G

def experiment_kirchhoff(G):
    try:
        rd = nx.resistance_distance(G)
        ki = sum(rd[i][j] for i in rd for j in rd[i]) / 2
    except TypeError:
        nodes = list(G.nodes)
        node_pairs = combinations(nodes, 2)
        ki = 0
        for nodeA, nodeB in node_pairs:
            rd = nx.resistance_distance(G, nodeA, nodeB)
            ki += rd
    return ki

def run_test():
    # Set up the graph
    G = setup_graph()
    
    # Measure the execution time of the resistance distance computation
    execution_time = timeit.timeit(lambda: experiment_kirchhoff(G), number=1)
    
    return execution_time
"""

TEST_HARNESS = TEST
TEST_HARNESS += """

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Measure performance of API.')
    parser.add_argument('output_file', type=str, help='File to append timing results to.')
    args = parser.parse_args()

    # Measure the execution time
    execution_time = run_test()

    # Append the results to the specified output file
    with open(args.output_file, 'a') as f:
        f.write(f'Execution time: {execution_time:.6f}s\\n')

if __name__ == '__main__':
    main()
"""
