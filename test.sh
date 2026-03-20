#!/bin/bash
set -e

MODE="${1:-cpu}"

case "$MODE" in
    cpu)
        IMAGE="evm-dev"
        DOCKERFILE="Dockerfile"
        TEST_FILE="tests/test_evm.py"
        ;;
    gpu)
        IMAGE="evm-cuda-dev"
        DOCKERFILE="Dockerfile.cuda"
        TEST_FILE="tests/test_evm_cuda.py"
        ;;
    --build)
        # ./test.sh --build  → force rebuild CPU image then test
        IMAGE="evm-dev"
        DOCKERFILE="Dockerfile"
        TEST_FILE="tests/test_evm.py"
        docker build \
            --build-arg UID="$(id -u)" \
            --build-arg GID="$(id -g)" \
            --build-arg UNAME="$(whoami)" \
            -f ${DOCKERFILE} -t ${IMAGE} .
        echo ""
        exec "$0" cpu
        ;;
    *)
        echo "Usage: ./test.sh [cpu|gpu|--build]"
        echo "  cpu     Run CPU unit tests (default)"
        echo "  gpu     Run GPU unit tests (requires NVIDIA GPU)"
        echo "  --build Force rebuild CPU image before testing"
        exit 1
        ;;
esac

# Build image if it doesn't exist
if ! docker image inspect ${IMAGE} &>/dev/null; then
    echo "Image ${IMAGE} not found. Building..."
    docker build \
        --build-arg UID="$(id -u)" \
        --build-arg GID="$(id -g)" \
        --build-arg UNAME="$(whoami)" \
        -f ${DOCKERFILE} -t ${IMAGE} .
    echo ""
fi

echo "=== Lint ==="
docker run --rm --entrypoint "" ${IMAGE} ruff check .

echo ""
echo "=== Tests (${MODE}) ==="
if [[ "$MODE" == "gpu" ]]; then
    docker run --rm --gpus all --entrypoint "" ${IMAGE} python3 -m pytest ${TEST_FILE} -v
else
    docker run --rm --entrypoint "" ${IMAGE} python -m pytest ${TEST_FILE} -v
fi

echo ""
echo "All checks passed."
