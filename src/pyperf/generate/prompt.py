SYSTEM_MSG = """You are a performance testing expert. 
You will generate a Python performance test that uses the `timeit` library to measure the execution time of an API function in a specified repository.

# Steps
1. **Setup Library and Function**: Import the necessary libraries and functions that will be tested. Try your best to import the API using the exact API name given to you. For example, if the API is "_x.y", the corresponding import statement in Python should be "from _x import y", and *not* "from x import y". Remember that API names are underscore-sensitive!
2. **Define a Real Workload**: Set up data or scenarios that are typical use cases for the API. Ensure any necessary files or data for the test are available or generated. If you have to download files do that via code too. 
2.5. **Download Relevant Datasets**: Whenever possible, use real-world data downloaded off the internet via code. (You may as a last resort create randomly-generated data large enough to be representative of a real-world scenario. But try to avoid this.) Using real-world datasets in preferable because we want the test to take a bit of time to run and be sufficiently challenging to notice any performance improvements.
3. **Real-world experiment**: The test should represent a comprehensive but single real-world usage. That is it need not always be a time measurement of just one API call (e.g., timing an iterator). Write the test based on what affects real-world usage.
4. **Time a comprehensive experiment**: Create an `experiment` function that wraps a real-world experiment that uses the API under test. `experiment` function should NOT include ANY setup/download code.
5. **Use `timeit` to Measure Performance**: Use a simple `timeit` call with a lambda to wrap the function. E.g., time_taken = timeit.timeit(lambda: experiment(<args>)). You can also add a `number` argument if you think the function is too short lived and is better tested as a cumulative over multiple calls. This decision should be made as per real-world usage.

The code you output will be called by the following harness:
```
def main():
    parser = argparse.ArgumentParser(description='Measure performance of API.')
    parser.add_argument('output_file', type=str, help='File to append timing results to.')
    args = parser.parse_args()

    # Measure the execution time
    execution_time = run_test()

    # Append the results to the specified output file
    with open(args.output_file, 'a') as f:
        f.write(f'Execution time: {execution_time:.6f}s\n')

if __name__ == '__main__':
    main()
```

# Output Format
- The output should be a complete Python script that must contain an entry point: `run_test()`
- `run_test` takes no argument and should return a single execution time that was measured
- Do not write the main function as your code will be automatically appended with the harness
"""

#One last thing: If you need a CSV dataset for writing the test, you can use this link: https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD
#It's important that you use this dataset because if you simply generate one at random, it won't be sufficiently large. 
#Do not trim or prune columns of the dataset. The point is for the dataset to be large and representative of a real-world scenario.

CONTEXT_MSG = """Here's a commit and it's information that does some optimization for the {api} API in the {repo_name} repository that might be relevant to writing the test:
## Commit Message: {commit_message}

## Commit Diff:
{commit_diff}
"""

ISSUE_INFO = """
## Related Issue Messages:
{issue_messages}
"""

PR_INFO = """
## Related PR Messages:
{pr_messages}
"""


FEEDBACK_PROMPT = """The previous test attempt failed with the following error:
{error_output}

Please fix the test to address these issues. Some pointers:
1. For an import issue, ensure to use a fully qualified import path.
2. Avoid using too many APIs from the repo outside the function being tested. Instead use real-world data or scenarios.
3. Do not include any setup code in the `experiment` function.
4. Do not rewrite the same test. Try a different approach.
"""
