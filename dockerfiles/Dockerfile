# override this to "nvidia/cuda:10.1-runtime-ubi8" if cuda is desired
ARG BASE_IMAGE=registry.access.redhat.com/ubi8/ubi:latest
FROM ${BASE_IMAGE}

ARG USER=genienlp

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

USER root

# copy source
COPY . /opt/genienlp

# install basic tools and python3, install dependencies, and then cleanup
# HACK: we need to download the wheel manually or docker will die
RUN dnf -y install git gcc gcc-c++ make cmake && \
    dnf -y module enable python38 \
        && dnf -y install python38 \
        python38-devel \
        python38-pip \
        python38-wheel \
	&& curl -sL https://files.pythonhosted.org/packages/1d/a9/f349273a0327fdf20a73188c9c3aa7dbce68f86fad422eadd366fd2ed7a0/torch-1.7.1-cp38-cp38-manylinux1_x86_64.whl \
	   -o torch-1.7.1-cp38-cp38-manylinux1_x86_64.whl \
	&& pip3 install torch-1.7.1-cp38-cp38-manylinux1_x86_64.whl \
	&& rm -fr torch-1.7.1-cp38-cp38-manylinux1_x86_64.whl \
	&& pip3 install kfserving \
        && pip3 install awscli \
	&& pip3 install -e /opt/genienlp \
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
