#!/bin/bash
set -e

VERSION=$(cat VERSION)-cuda

docker build \
    -f Dockerfile.cuda \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    --build-arg VERSION="${VERSION}" \
    -t evm-cuda:${VERSION} \
    -t evm-cuda:latest .

echo "Built evm-cuda:${VERSION} (also tagged :latest)"
echo "Run with: docker run --gpus all --rm -v \"\$(pwd)\":/data evm-cuda -i /data/input.mp4 -o /data/output.avi"
