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
docker build -t routing .

# Remove any existing container with the same name (optional)
docker container rm routing1 || true

# Run the new container with the environment variable
docker run -d -p 5002:5002 --env GOOGLE_MAP_API_KEY="$GOOGLE_MAP_API_KEY" routing

