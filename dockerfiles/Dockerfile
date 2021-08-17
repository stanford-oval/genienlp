# override this to "nvidia/cuda:10.1-runtime-ubi8" if cuda is desired
ARG BASE_IMAGE=932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:base
FROM ${BASE_IMAGE}

ARG USER=genienlp

LABEL maintainer="Thingpedia Admins <thingpedia-admins@lists.stanford.edu>"

USER root

# copy source
COPY . /opt/genienlp

# install basic tools and python3, install dependencies, and then cleanup
# HACK: we need to download the wheel manually or docker will die
RUN dnf -y install \
        git \
        gcc \
        gcc-c++ \
        make \
        cmake \
    && dnf -y module enable python38 \
    && dnf -y install \
        python38 \
        python38-devel \
        python38-pip \
        python38-wheel \
    && pip3 install --upgrade pip \
    && pip3 install \
        awscli \
        -e /opt/genienlp \
    && python3 -m spacy download en_core_web_sm \
	&& rm -fr /root/.cache \
	&& dnf -y remove gcc gcc-c++ make cmake \
	&& rm -fr /var/cache/dnf


# add user genienlp
RUN useradd -ms /bin/bash -r genienlp

# User root is required for kfserving until https://github.com/aws/amazon-eks-pod-identity-webhook/issues/8 is fixed.
# There is a workaround by changing pod fsgroup but kfserving inference service cannot set pod level security context.
USER ${USER}

WORKDIR /home/genienlp
ENTRYPOINT ["/opt/genienlp/dockerfiles/start.sh"]
