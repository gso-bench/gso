#!/bin/bash
set -e

# Parameters:
# $1 - Namespace (defaults to 'slimshetty') or number of workers if it's a number
# $2 - Number of parallel workers (defaults to 4) if namespace was provided
NAMESPACE="slimshetty"
WORKERS=4

# Check if first argument is a number (indicating workers)
if [[ $1 =~ ^[0-9]+$ ]]; then
    WORKERS=$1
else
    # If not a number, use it as namespace if provided
    if [ ! -z "$1" ]; then
        NAMESPACE=$1
    fi

    # Check if second argument exists and use it as workers
    if [ ! -z "$2" ]; then
        WORKERS=$2
    fi
fi

echo "Using namespace: $NAMESPACE"
echo "Using parallel workers: $WORKERS"

IMAGE_FILE="$(dirname "$0")/all-gso-instance-images.txt"
PATTERN="gso.eval"

echo "Pulling docker images"
echo "Pattern: $PATTERN"
echo "Image file: $IMAGE_FILE"

# Create a temporary file to store the filtered images
TEMP_FILE=$(mktemp)
grep "$PATTERN" "$IMAGE_FILE" > "$TEMP_FILE"

# Count total images to pull
TOTAL_IMAGES=$(wc -l < "$TEMP_FILE")
echo "Found $TOTAL_IMAGES images to pull"

# Function to pull images from a subset of lines
pull_images() {
    local start=$1
    local step=$2
    local worker_id=$3

    echo "Worker $worker_id starting"

    for (( i=start; i<TOTAL_IMAGES; i+=step )); do
        # Get the image at line number i+1 (because sed is 1-indexed)
        image=$(sed -n "$((i+1))p" "$TEMP_FILE")
        echo "Worker $worker_id pulling $NAMESPACE/$image"
        docker pull "$NAMESPACE/$image" && echo "Worker $worker_id completed: $image" || echo "Worker $worker_id failed: $image"
    done

    echo "Worker $worker_id finished"
}

# Launch workers in parallel
for (( worker=0; worker<WORKERS; worker++ )); do
    pull_images $worker $WORKERS $worker &
done

# Wait for all background processes to finish
wait

echo "All docker pulls completed"

# Clean up
rm "$TEMP_FILE"