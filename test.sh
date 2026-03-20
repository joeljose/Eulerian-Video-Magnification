#!/bin/bash
set -e

IMAGE="evm-dev"

# Build only if image doesn't exist or --build flag passed
if [[ "$1" == "--build" ]] || ! docker image inspect ${IMAGE} &>/dev/null; then
    echo "Building test image..."
    docker build \
        --build-arg UID="$(id -u)" \
        --build-arg GID="$(id -g)" \
        --build-arg UNAME="$(whoami)" \
        -t ${IMAGE} .
    echo ""
fi

echo "=== Lint ==="
docker run --rm --entrypoint "" ${IMAGE} ruff check .

echo ""
echo "=== Tests ==="
docker run --rm --entrypoint "" ${IMAGE} python -m pytest tests/ -v

echo ""
echo "All checks passed."
