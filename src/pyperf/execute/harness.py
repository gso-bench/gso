TEST = """
import timeit
import numpy as np
import pandas as pd
from datasets import Dataset

def generate_large_dataset(num_rows=1000, num_features=6000):
    # Generate a large dataset with specified number of rows and features.
    data = {f'feature_{i}': np.random.rand(num_rows) for i in range(num_features)}
    return Dataset.from_pandas(pd.DataFrame(data))

def map_function(batch):
    return batch

def experiment(ds):
    # Perform the experiment by applying the map function to the dataset.
    # Apply the map function
    ds.map(map_function, batched=True)

def run_test():
    # Generate a large dataset
    ds = generate_large_dataset()
    
    # Run the performance test and return the execution time.
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
