#!/usr/bin/env bash

# Common / useful `set` commands
set -Ee # Exit on error
set -o pipefail # Check status of piped commands
set -u # Error on undefined vars
# set -v # Print everything
# set -x # Print commands (with expanded vars)

cd "$(git rev-parse --show-toplevel)"

echo "Resetting pyenv virutalenv..."

if [[ -f ./.python-version ]]; then
	echo "Reading pyenv virtualenv version and name from .python-version..."
	python_version="$(cat ./.python-version)"
	version="${python_version%%/*}"
	name="${python_version##*/}"
else
	name="genienlp"
	version="3.8.11" # Latest 3.8 as of 2021-08-13
	python_version="${version}/envs/${name}"
fi

echo "python version:  ${version}"
echo "virtualenv name: ${name}"

echo "Deleting virtualenv ${name} (if present)"
pyenv virtualenv-delete --force "${name}" || true # --force doesn't work?!?

echo "Creting virtualenv ${version} ${name}"
pyenv virtualenv "${version}" "${name}"

echo "Writing \"${python_version}\" -> .python-version"
echo "${python_version}" > ./.python-version

echo "Upgrading pip"
"$(pyenv prefix)/bin/pip" install --upgrade pip

echo "pyenv virtualenv ${name} reset."
