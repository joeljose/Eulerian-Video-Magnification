#!/bin/bash
set -e

docker build \
    -f Dockerfile.cuda \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    -t evm-cuda .

echo "Built evm-cuda image as user: $(whoami) (uid=$(id -u), gid=$(id -g))"
echo "Run with: docker run --gpus all --rm -v \"\$(pwd)\":/data evm-cuda -i /data/input.mp4 -o /data/output.avi"
