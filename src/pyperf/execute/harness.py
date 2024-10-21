TEST = """
import networkx as nx
import timeit

def setup_graph(num_nodes):
    # Set up a graph with a specified number of nodes.
    # Create a random graph with a specified number of nodes and a probability for edge creation
    G = nx.erdos_renyi_graph(num_nodes, 0.1, seed=42)
    return G

def experiment(G):
    # Experiment to measure the execution time of common neighbor centrality.
    # Compute common neighbor centrality for all pairs of nodes
    centrality = nx.common_neighbor_centrality(G)
    # Consume the generator to ensure the computation is complete
    list(centrality)

def run_test():
    # Set up the graph (do not time this part)
    num_nodes = 1000
    G = setup_graph(num_nodes)

    # Use timeit to measure the performance of the experiment
    time_taken = timeit.timeit(lambda: experiment(G), number=1)
    
    # Return the time taken for the experiment
    return time_taken
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
