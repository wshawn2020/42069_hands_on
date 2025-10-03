#!/bin/sh

# Author: Shawn Wang
# Email: wshawn2020@gmail.com
# Github: https://github.com/wshawn2020

# Build the Docker image:
echo "[1/7] Build demo image my-python-auto"
docker build -t my-python-auto .

echo "[2/7] Show built image"
docker images

# Build the Container:
echo "[3/7] Create demo container my-auto-container and run random forest detection"
docker-compose up -d

echo "[4/7] Show built container"
docker ps -a

# Detect threats
echo "[5/7] Display detection process running in containerized environment"
docker logs my-auto-container -f

echo "[6/7] Stopping docker containers"
docker-compose down

# Finish
echo "[7/7] Demo finished"

