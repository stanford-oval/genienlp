# override this to "nvidia/cuda:10.1-runtime-ubi7" if cuda is desired
ARG BASE_IMAGE=registry.access.redhat.com/ubi7/ubi:latest
FROM ${BASE_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

# install basic tools
RUN yum -y install git wget gcc gcc-c++

# add epel repo
RUN wget http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm -O /tmp/epel-release-latest-7.noarch.rpm
RUN yum -y install /tmp/epel-release-latest-7.noarch.rpm

# install python3
RUN yum -y install python36 \
	python36-numpy \
	python36-scipy \
	python36-pip \
	python36-devel

# add user decanlp
RUN useradd -ms /bin/bash -r decanlp
USER decanlp
WORKDIR /home/decanlp

# download embeddings
RUN mkdir -p .embeddings
WORKDIR .embeddings
RUN for v in charNgram glove.840B.300d ; do \
		for f in vectors table itos ; do \
			wget -c https://oval.cs.stanford.edu/data/glove/${v}.txt.${f}.npy ; \
		done ; \
	done

# copy source and install
USER root
COPY . /opt/decanlp
RUN pip3 install -e /opt/decanlp

USER decanlp
WORKDIR /home/decanlp
ENTRYPOINT ["decanlp"]
