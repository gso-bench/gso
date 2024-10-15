TEST_HARNESS = """
{test}

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
""".format(
    test=TEST
)


TEST = """
import timeit
from PIL import Image
import os
import requests

def download_sample_image(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def setup_image():
    image_url = 'https://via.placeholder.com/800x600.png'
    image_filename = 'sample_image.png'
    
    if not os.path.exists(image_filename):
        download_sample_image(image_url, image_filename)
    
    return image_filename

def experiment(image_filename):
    with Image.open(image_filename) as img:
        # Save the image to a new file
        img.save('output_image.png')

def run_test():
    image_filename = setup_image()
    
    # Measure the execution time of the experiment
    execution_time = timeit.timeit(lambda: experiment(image_filename), number=10)
    
    # Clean up the output image file
    if os.path.exists('output_image.png'):
        os.remove('output_image.png')
    
    return execution_time
"""
