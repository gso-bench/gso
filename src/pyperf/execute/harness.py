TEST = """
import timeit
from PIL import Image
import requests
from io import BytesIO

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def experiment(image):
    # Convert to a format that supports transparency (RGBA)
    image = image.convert("RGBA")

    # Use the getbbox function to find the bounding box of the non-transparent part
    bbox = image.getbbox()
    return bbox

def run_test():
    # Download a real-world image
    url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    image = download_image(url)
    
    # Measure the execution time using timeit
    execution_time = timeit.timeit(lambda: experiment(image), number=100)
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
