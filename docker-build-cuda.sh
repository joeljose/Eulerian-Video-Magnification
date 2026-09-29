#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    -f Dockerfile.cuda \
    --build-arg VERSION="${VERSION}" \
    -t evm-cuda:${VERSION} \
    -t evm-cuda:latest .

echo "Built evm-cuda:${VERSION} (also tagged :latest)"
echo "Run with: docker run --gpus all --rm -v \"\$(pwd)\":/data evm-cuda -i /data/input.mp4 -o /data/output.avi"
