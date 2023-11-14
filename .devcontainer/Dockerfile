# This a development container for VSCode
# To create a production container, I recommend this article:
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ARG USERNAME=pablo
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
    tree \
    && rm -rf /var/lib/apt/lists/*

# Create a new user in the container with the same UID as the host user
RUN useradd -m -u $USER_UID $USERNAME

# Change ownership of the Conda site-packages directory
RUN chown -R $USER_UID:$USER_UID /opt/conda/lib/python3.10/site-packages/

# Set the user as the default user for the container
# USER $USERNAME

WORKDIR /home/$USERNAME

# Add poetry to PATH
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

USER root

# Install poetry
RUN pip install poetry==1.7.0  --no-cache-dir

# Install poetry dependencies
COPY --chown=$USER_UID:$USER_UID pyproject.toml poetry.lock ./

# Install poetry dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi 

USER $USERNAME
