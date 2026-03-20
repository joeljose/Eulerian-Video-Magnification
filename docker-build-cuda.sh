#!/bin/bash
set -e

VERSION=$(python3 -c "exec(open('evm_cuda.py').read().split('__version__')[0]+'__version__'+open('evm_cuda.py').read().split('__version__')[1].split('\n')[0]); print(__version__)" 2>/dev/null || grep -oP '__version__ = "\K[^"]+' evm_cuda.py)

docker build \
    -f Dockerfile.cuda \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    -t evm-cuda:${VERSION} \
    -t evm-cuda:latest .

echo "Built evm-cuda:${VERSION} (also tagged :latest)"
echo "Run with: docker run --gpus all --rm -v \"\$(pwd)\":/data evm-cuda -i /data/input.mp4 -o /data/output.avi"
