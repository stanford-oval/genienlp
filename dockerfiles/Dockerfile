# override this to "nvidia/cuda:10.1-runtime-ubi8" if cuda is desired
ARG BASE_IMAGE=registry.access.redhat.com/ubi8/ubi:latest
FROM ${BASE_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

# install basic tools and python3
RUN dnf -y install git wget gcc gcc-c++ \
	python3 \
	python3-numpy \
	python3-scipy \
	python3-pip \
	python3-devel \
	&& rm -fr /var/cache/dnf

# download embeddings
RUN mkdir -p /usr/local/share/decanlp/embeddings
WORKDIR /usr/local/share/decanlp/embeddings
RUN for v in charNgram glove.840B.300d ; do \
		for f in vectors table itos ; do \
			wget -c https://oval.cs.stanford.edu/data/glove/${v}.txt.${f}.npy ; \
		done ; \
	done

# add user decanlp
RUN useradd -ms /bin/bash -r decanlp
USER decanlp
WORKDIR /home/decanlp
RUN ln -s /usr/local/share/decanlp/embeddings /home/decanlp/.embeddings

# copy source and install
USER root
COPY . /opt/decanlp
RUN pip3 install -e /opt/decanlp && rm -fr /root/.cache

USER decanlp
WORKDIR /home/decanlp
ENTRYPOINT ["decanlp"]
