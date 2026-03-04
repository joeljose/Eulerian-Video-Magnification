#!/bin/bash
set -e

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    -t eulerian-video-magnification .

echo "Built eulerian-video-magnification image as user: $(whoami) (uid=$(id -u), gid=$(id -g))"
