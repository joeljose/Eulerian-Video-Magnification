#!/bin/bash
set -e

VERSION=$(grep -oP '__version__ = "\K[^"]+' evm.py)

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    -t evm:${VERSION} \
    -t evm:latest .

echo "Built evm:${VERSION} (also tagged :latest)"
