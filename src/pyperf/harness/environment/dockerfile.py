DOCKERFILE = r"""
FROM --platform={platform} ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y \
wget \
git \
build-essential \
libffi-dev \
libtiff-dev \
python3 \
python3-pip \
python-is-python3 \
jq \
curl \
locales \
locales-all \
tzdata \
# Add packages BLAS/LAPACK support
gcc \
g++ \
gfortran \
libopenblas-dev \
liblapack-dev \
pkg-config \
# Add image processing libs
libtiff5-dev \
libjpeg8-dev \
libopenjp2-7-dev \
zlib1g-dev \
libfreetype6-dev \
liblcms2-dev \
libwebp-dev \
tcl8.6-dev \
tk8.6-dev \
python3-tk \
libharfbuzz-dev \
libfribidi-dev \
libxcb1-dev \
libx11-dev \
&& rm -rf /var/lib/apt/lists/*

# Download and install conda
RUN wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-{conda_arch}.sh' -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda3
# Add conda to PATH
ENV PATH=/opt/miniconda3/bin:$PATH
# Add conda to shell startup scripts like .bashrc (DO NOT REMOVE THIS)
RUN conda init --all
RUN conda config --append channels conda-forge

# Download and instal uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN . $HOME/.local/bin/env
# Add uv to PATH
ENV PATH=/root/.local/bin:$PATH

RUN adduser --disabled-password --gecos 'dog' nonroot

WORKDIR /testbed/

# Copy and setup the repo
COPY ./setup_repo.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_repo.sh
RUN /bin/bash /root/setup_repo.sh

# Copy the test and eval scripts to parent of testbed
COPY ./eval.sh /
COPY ./pyperf_test*.py /

WORKDIR /testbed/

# Automatically activate the env within the testbed directory
RUN echo "source .venv/bin/activate" >> /root/.bashrc
"""


def get_dockerfile_instance(platform: str, arch: str) -> str:
    if arch == "arm64":
        conda_arch = "aarch64"
    else:
        conda_arch = arch
    return DOCKERFILE.format(platform=platform, conda_arch=conda_arch)
