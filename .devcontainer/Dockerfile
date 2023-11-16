# This a development container for VSCode
# To create a production container, I recommend this article:
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0
# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ARG USERNAME=pablo
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
# Install system packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3-distutils \
    build-essential \
    curl \
    git \
    openssh-client \
    iproute2 \
    procps \
    lsb-release \
    sudo \
    wget \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as the default python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --set python /usr/bin/python3.11

# Create a new user in the container with the same UID as the host user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -m -s /bin/bash --uid $USER_UID --gid $USER_GID $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
    
# Change to non-root privilege
USER $USERNAME

# Install pip as non-root user
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Set the working directory
WORKDIR /home/$USERNAME

# Add local bin to PATH
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Install poetry
RUN pip install poetry==1.7.0  --no-cache-dir

# Copy the Python project files
COPY --chown=$USER_UID:$USER_GID pyproject.toml poetry.lock ./

# Install Python dependencies using Poetry
RUN poetry install --no-interaction --no-ansi
