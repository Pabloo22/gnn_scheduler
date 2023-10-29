# Use an official Python as a parent image
FROM python:3.11 AS build

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only requirements to cache them in the docker layer
WORKDIR /alpha_graph
COPY pyproject.toml poetry.lock ./

# Generate requirements.txt from Poetry configuration
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

# Start a new stage for a cleaner image
FROM python:3.11 AS release

# Metadata
LABEL maintainer="pablo.arino@alumnos.upm.es"
LABEL version="1.0"
LABEL description="Python image for GNN development with Poetry"

# Create and switch to a non-root user and prepare for volume mount
RUN useradd --create-home newuser
USER newuser
WORKDIR /home/newuser

# Copy requirements and install packages
COPY --from=build /alpha_graph/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prepare for volume mount
RUN chown -R newuser:newuser /home/newuser
