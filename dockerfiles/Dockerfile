# override this to "nvidia/cuda:10.1-runtime-ubi8" if cuda is desired
ARG BASE_IMAGE=registry.access.redhat.com/ubi8/ubi:latest
FROM ${BASE_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

USER root

# copy source
COPY . /opt/genienlp

# install basic tools and python3, install dependencies, and then cleanup
RUN dnf -y install git gcc gcc-c++ make cmake && \
    dnf -y module enable python38 \
        && dnf -y install python38 \
        python38-devel \
        python38-pip \
        python38-wheel \
        && pip3 install --upgrade pip \
        && pip3 install --use-feature=2020-resolver awscli \
	&& pip3 install --use-feature=2020-resolver -e /opt/genienlp \
	&& python3 -m spacy download en_core_web_sm \
	&& rm -fr /root/.cache \
	&& dnf -y remove gcc gcc-c++ make cmake \
	&& rm -fr /var/cache/dnf


# add user genienlp
RUN useradd -ms /bin/bash -r genienlp
USER genienlp

WORKDIR /home/genienlp
ENTRYPOINT ["/opt/genienlp/dockerfiles/start.sh"]
