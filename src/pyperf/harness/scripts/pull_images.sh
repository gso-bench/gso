#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -r REPOSITORY [-i IMAGE_NAMES] [-h]"
    echo
    echo "Pull Docker images from Docker Hub repository"
    echo
    echo "Options:"
    echo "  -r REPOSITORY    Docker Hub repository (e.g., slimshetty/pyperf) (required)"
    echo "  -i IMAGE_NAMES   Comma-separated list of image tags (optional)"
    echo "  -s              Create short aliases without repository prefix (optional)"
    echo "  -h              Display this help message"
    echo
    echo "Examples:"
    echo "  $0 -r slimshetty/pyperf -s"
    echo "  $0 -r slimshetty/pyperf -i \"latest,1.0.0,dev\" -s"
    exit 1
}

# Parse command line arguments
while getopts "r:i:sh" opt; do
    case $opt in
        r) REPO="$OPTARG";;
        i) TAGS="$OPTARG";;
        s) CREATE_SHORT_ALIASES=true;;
        h) usage;;
        ?) usage;;
    esac
done

# Check if repository is provided
if [ -z "$REPO" ]; then
    echo "Error: Docker Hub repository is required"
    usage
fi

# Validate repository format (should contain a slash)
if [[ ! "$REPO" =~ "/" ]]; then
    echo "Error: Invalid repository format. Expected format: username/repository"
    usage
fi

# If no specific tags are provided, get all tags from Docker Hub
if [ -z "$TAGS" ]; then
    echo "Fetching all tags for $REPO"
    TAGS=""
    NEXT_URL="https://hub.docker.com/v2/repositories/$REPO/tags?page_size=100"
    
    while [ -n "$NEXT_URL" ]; do
        RESPONSE=$(curl -s "$NEXT_URL")
        PAGE_TAGS=$(echo "$RESPONSE" | grep -o '"name":"[^"]*' | grep -o '[^"]*$' | tr '\n' ',' | sed 's/,$//')
        TAGS="$TAGS,$PAGE_TAGS"
        NEXT_URL=$(echo "$RESPONSE" | grep -o '"next":"[^"]*' | grep -o 'http[^"]*' || echo "")
    done
    
    TAGS=$(echo "$TAGS" | sed 's/^,//')
    
    if [ -z "$TAGS" ]; then
        echo "Error: No tags found for repository $REPO"
        exit 1
    fi
fi

# Pull images and create aliases if requested
echo "Pulling images from $REPO"
IFS=',' read -ra TAG_ARRAY <<< "$TAGS"
for tag in "${TAG_ARRAY[@]}"; do
    # Trim whitespace
    tag=$(echo "$tag" | xargs)
    echo "Pulling $REPO:$tag"
    docker pull "$REPO:$tag"
    
    # Create short alias if requested
    if [ "$CREATE_SHORT_ALIASES" = true ]; then
        echo "Creating short alias: $tag"
        docker tag "$REPO:$tag" "$tag"
        echo "You can now use '$tag' instead of '$REPO:$tag'"
    fi
done

echo "Image pull complete!"
if [ "$CREATE_SHORT_ALIASES" = true ]; then
    echo "Short aliases have been created. You can use them directly in your Python code."
    echo "Example Python usage:"
    echo "    client.containers.run('$tag', ...)"
fi