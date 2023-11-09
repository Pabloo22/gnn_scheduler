# Start from the PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set a working directory
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files (if available) into the container
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
# We avoid creating a virtual environment inside the Docker container
# by setting the `POETRY_VIRTUALENVS_CREATE` environment variable to false.
# This is because the container, by its nature, is already an isolated environment.
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# Install dependencies using Poetry
# This step will install the dependencies specified in the pyproject.toml file
RUN poetry install

# If you also want to copy your source code into the container, uncomment the following line
COPY . /app
