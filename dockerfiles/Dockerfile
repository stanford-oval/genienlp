# override this to "nvidia/cuda:10.1-runtime-ubi8" if cuda is desired
ARG BASE_IMAGE=registry.access.redhat.com/ubi8/ubi:latest
FROM ${BASE_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

USER root

# copy source
COPY . /opt/decanlp

# install basic tools and python3, install dependencies, and then cleanup
RUN dnf -y install git gcc gcc-c++ \
	python3 \
	python3-numpy \
	python3-scipy \
	python3-pip \
	python3-devel \
	&& pip3 install -e /opt/decanlp \
	&& rm -fr /root/.cache \
	&& dnf -y remove gcc gcc-c++ \
	&& rm -fr /var/cache/dnf

# add user decanlp
RUN useradd -ms /bin/bash -r decanlp
USER decanlp
WORKDIR /home/decanlp
ENTRYPOINT ["decanlp"]
