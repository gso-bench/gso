TEST = """
import timeit
from PIL import Image
import requests
from io import BytesIO

def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def setup_images():
    # Download a sample image with alpha channel
    url1 = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    image1 = download_image(url1).convert("RGBA")
    
    # Create a simple RGBA image programmatically
    image2 = Image.new("RGBA", image1.size, (255, 0, 0, 128))  # Semi-transparent red

    return image1, image2

def experiment(image1, image2):
    # Perform alpha compositing
    result = Image.alpha_composite(image1, image2)
    return result

def run_test():
    # Setup the images
    image1, image2 = setup_images()

    # Measure the execution time of the experiment
    execution_time = timeit.timeit(lambda: experiment(image1, image2), number=100)
    
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
