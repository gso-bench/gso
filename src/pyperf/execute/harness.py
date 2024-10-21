TEST = """
import timeit
from datasets import load_dataset

def setup():
    # Load the dataset outside the experiment function to avoid timing the setup
    dataset = load_dataset('stanfordnlp/imdb', split='train')
    return dataset

def experiment(ds):
    # Define a filter function that simulates a real-world use case
    def filter_function(example):
        return len(example['text']) > 100

    # Apply the filter function to the dataset
    filtered_dataset = ds.filter(filter_function)

def run_test():
    ds = setup()
    
    # Measure the execution time of the experiment
    execution_time = timeit.timeit(lambda: experiment(ds), number=1)
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
