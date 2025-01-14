#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 565393027942.dkr.ecr.ap-southeast-2.amazonaws.com

echo "Pulling Docker image..."
docker 565393027942.dkr.ecr.ap-southeast-2.amazonaws.com/yt-sentiment-analysis:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=amit-app)" ]; then
    echo "Stopping existing container..."
    docker stop campusx-app
fi

if [ "$(docker ps -aq -f name=amit-app)" ]; then
    echo "Removing existing container..."
    docker rm campusx-app
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name amit-app 565393027942.dkr.ecr.ap-southeast-2.amazonaws.com/yt-sentiment-analysis:latest

echo "Container started successfully."