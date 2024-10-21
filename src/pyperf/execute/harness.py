TEST = """
import timeit
import networkx as nx
import urllib.request
import os
import gzip

def download_and_extract_dataset():
    url = "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz"
    filename = "com-youtube.ungraph.txt.gz"
    extracted_filename = "com-youtube.ungraph.txt"

    # Download the dataset if it doesn't exist
    if not os.path.exists(filename):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, filename)
    
    # Extract the dataset if it hasn't been extracted
    if not os.path.exists(extracted_filename):
        print("Extracting dataset...")
        with gzip.open(filename, 'rb') as f_in:
            with open(extracted_filename, 'wb') as f_out:
                f_out.write(f_in.read())
    
    return extracted_filename

def setup_real_world_graph():
    # Download and extract the dataset
    dataset_file = download_and_extract_dataset()

    # Load the graph from the dataset
    print("Loading graph...")
    G = nx.Graph()
    with open(dataset_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v)
    
    return G

def experiment(graph):
    # Measure the time taken to find all connected components in the graph
    components = list(nx.connected_components(graph))
    return components

def run_test():
    # Setup the graph outside of the timing function to focus on the API call
    graph = setup_real_world_graph()

    # Use timeit to measure the execution time of the experiment
    time_taken = timeit.timeit(lambda: experiment(graph), number=20)
    
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
