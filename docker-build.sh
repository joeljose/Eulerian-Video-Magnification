#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    --build-arg VERSION="${VERSION}" \
    -t evm:${VERSION} \
    -t evm:latest .

echo "Built evm:${VERSION} (also tagged :latest)"
