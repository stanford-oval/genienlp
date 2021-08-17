#!/usr/bin/env bash

# Common / useful `set` commands
set -Ee # Exit on error
set -o pipefail # Check status of piped commands
set -u # Error on undefined vars
# set -v # Print everything
# set -x # Print commands (with expanded vars)

cd "$(git rev-parse --show-toplevel)"

docker build \
	--build-arg BASE_IMAGE=nvidia/cuda:10.1-runtime-ubi8 \
	-f ./dockerfiles/Dockerfile \
	-t "stanfordoval/genienlp:$(git describe)-cuda" \
	.