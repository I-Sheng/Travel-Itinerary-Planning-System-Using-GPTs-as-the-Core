#!/bin/bash

set -e

# Get the current directory
current_dir=$(pwd)

# Ensure .env file exists
if [ ! -f "$current_dir/../.env" ]; then
    echo ".env file not found in parent directory."
    exit 1
fi

# Source environment variables
source "$current_dir/../.env"

# Ensure sitesData.json exists
if [ ! -f "$current_dir/../data/sitesData.json" ]; then
    echo "sitesData.json not found in parent directory."
    exit 1
fi

# Ensure ./data directory exists
mkdir -p ./data

# Copy sitesData.json to the current directory
cp "$current_dir/../data/sitesData.json" ./data

# Build the Docker image
docker build -t recommend .

# Remove any existing container with the same name (optional)
docker rm -f recommend || true

# Run the new container with the environment variable
docker run -d -p 5001:5001 --env OPENAI_API_KEY=$OPENAI_API_KEY recommend

