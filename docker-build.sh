#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg VERSION="${VERSION}" \
    -t evm:${VERSION} \
    -t evm:latest .

echo "Built evm:${VERSION} (also tagged :latest)"
