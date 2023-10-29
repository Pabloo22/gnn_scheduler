# Use an official Python as a parent image
FROM python:3.11 AS build

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only requirements to cache them in the docker layer
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install dependencies and export them to a requirements.txt file
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

# Start a new stage for a cleaner image
FROM python:3.11 AS release

# Metadata
LABEL maintainer="pablo.arino@alumnos.upm.es"
LABEL version="1.0"
LABEL description="Python image for GNN development with Poetry"

# Create and switch to a non-root user
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# Copy requirements and install packages
COPY --from=build /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sample entrypoint script to execute python script or run other commands
# COPY entrypoint.sh .
# RUN chmod +x ./entrypoint.sh
# ENTRYPOINT ["./entrypoint.sh"]

# Uncomment if you need CMD instead
# CMD ["python", "./your-script.py"]
