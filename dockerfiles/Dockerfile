# override this to "nvidia/cuda:10.1-runtime-ubi8" if cuda is desired
ARG BASE_IMAGE=registry.access.redhat.com/ubi8/ubi:latest
FROM ${BASE_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

USER root

# copy source
COPY . /opt/genienlp

# install basic tools and python3, install dependencies, and then cleanup
# HACK: we need to download the wheel manually or docker will die
RUN dnf -y install git gcc gcc-c++ \
	python3 \
	python3-numpy \
	python3-scipy \
	python3-pip \
	python3-devel \
	&& curl -sL https://files.pythonhosted.org/packages/24/19/4804aea17cd136f1705a5e98a00618cb8f6ccc375ad8bfa437408e09d058/torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl \
	   -o torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl \
	&& pip3 install torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl \
	&& rm -fr torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl \
	&& pip3 install -e /opt/genienlp \
	&& rm -fr /root/.cache \
	&& dnf -y remove gcc gcc-c++ \
	&& rm -fr /var/cache/dnf

# add user genienlp
RUN useradd -ms /bin/bash -r genienlp
USER genienlp
WORKDIR /home/genienlp
ENTRYPOINT ["genienlp"]
