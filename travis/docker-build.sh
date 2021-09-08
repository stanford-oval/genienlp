#!/bin/bash

set -e
set -x

cd dockerfiles

export DOCKERFILE_PATH=dockerfiles/Dockerfile
export DOCKER_REPO=stanfordoval/genienlp

export IMAGE_NAME=stanfordoval/genienlp:latest
docker pull $IMAGE_NAME
./hooks/build
docker push $IMAGE_NAME
./hooks/post_push

export IMAGE_NAME=stanfordoval/genienlp:latest-cuda
docker pull $IMAGE_NAME
./hooks/build
docker push $IMAGE_NAME
./hooks/post_push
