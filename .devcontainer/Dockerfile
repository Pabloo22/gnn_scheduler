FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ARG USERNAME=vscode
ARG USER_UID=1000

# Install system packages
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    curl \
    git \
    openssh-client \
    iproute2 \
    procps \
    lsb-release \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a new user in the container with the same UID as the host user
RUN useradd -m -u $USER_UID $USERNAME

# Set the user as the default user for the container
USER $USERNAME

# Install poetry
RUN pip install poetry==1.7.0  --no-cache-dir
