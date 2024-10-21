TEST = """
import timeit
import networkx as nx
import random

def setup_graph():
    N = 2000
    random.seed(42)
    G = nx.complete_graph(N)
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.random()
    return G

def experiment(G):
    list(nx.minimum_spanning_edges(G, algorithm="prim", weight='weight', data=True))

def run_test():
    G = setup_graph()

    # Measure the execution time using timeit
    time_taken = timeit.timeit(lambda: experiment(G), number=1)
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
