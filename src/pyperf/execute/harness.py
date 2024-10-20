TEST = """
import os
import timeit
from datasets import Dataset
import numpy as np
from tqdm.auto import tqdm

def setup():
    # Create a large synthetic dataset
    size = 10_000_000
    ds = Dataset.from_dict({
        "id": range(size),
        "text": [f"This is sample text number {i}" for i in range(size)],
        "value": np.random.rand(size).tolist()
    })
    
    # Ensure the temporary directory exists
    os.makedirs("tmp", exist_ok=True)
    
    return ds
    
def experiment(ds):
    num_shards = 5

    for index in tqdm(range(num_shards), desc="Sharding"):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard.to_json(f"tmp/data_original_{index}.jsonl")

        
def run_test():
    # Setup the experiment
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
