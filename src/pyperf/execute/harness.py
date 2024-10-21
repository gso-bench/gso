TEST = """
import timeit
import networkx as nx

def setup_graph():
    # Set up a directed graph with a large number of nodes.
    g = nx.DiGraph()
    for i in range(10_000):
        g.add_node(i)
    return nx.subgraph_view(g, filter_node=lambda n: True, filter_edge=lambda u, v: True)

def experiment(graph):
    # Run the weakly_connected_components function on the graph.
    for _ in nx.weakly_connected_components(graph):
        pass

def run_test():
    # Measure the execution time of the weakly_connected_components function.
    graph = setup_graph()
    # Measure the execution time using timeit
    execution_time = timeit.timeit(lambda: experiment(graph), number=1)
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
