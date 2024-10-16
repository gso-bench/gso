TEST = """
import timeit
from PIL import Image
from io import BytesIO

def create_test_image():
    # Create a large image with a palette that requires optimization
    # Using a 9500x4000 image with ~135 colors as per the commit message
    width, height = 9500, 4000
    colors = 135
    # Create an image with a gradient to simulate real-world data
    im = Image.new("P", (width, height))
    palette = [i % 256 for i in range(3 * 256)]
    im.putpalette(palette)
    for x in range(width):
        for y in range(height):
            im.putpixel((x, y), (x * y) % colors)
    return im

def experiment(im):    
    # Save the image to a BytesIO object to simulate file saving
    output = BytesIO()
    im.save(output, format='GIF', optimize=True)

def run_test():
    # Create the test image
    im = create_test_image()
    
    # Measure the execution time of the experiment
    execution_time = timeit.timeit(lambda: experiment(im), number=1)
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
