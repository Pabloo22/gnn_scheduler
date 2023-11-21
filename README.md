# Solving the Job-Shop Scheduling Problem with Graph Neural Networks
This repository contains all the code related to solving the Job-Shop Scheduling 
Problem (JSSP) with Graph Neural Networks (GNNs). 

For a step-by-step explanation of the source code of this project see the [Tutorial](notebooks/tutorial).

## Installation

### Visual Studio Code and Dev Containers
A `.devcontainer` folder in the repository is included. This is set to work with VisualStudio Code and allows to easily work inside a Docker container and avoid specific machines used. This is the option recommended if you plan to contribute to this repository. For more information about devcontainers click [here](https://code.visualstudio.com/docs/devcontainers/containers).

#### Prerequisites:

1. **Docker Installed:** Ensure Docker is installed on your system. For GPU support, you need the community edition (CE).
2. **NVIDIA GPU Drivers:** Install the latest NVIDIA drivers for your GPU from the official website.
3. **NVIDIA Docker Toolkit:** Install the NVIDIA Container Toolkit, which allows Docker to use the GPU.


### Poetry
For a simpler installation of the project, [Poetry](https://python-poetry.org/) is recommended, which is a tool for dependency management and packaging in Python. It helps to create and manage virtual environments, streamlining the process of setting up and maintaining project dependencies. Poetry automatically handles dependency resolution and ensures that your project environment is consistent. 

#### Prerequisites:
- **Python 3.11.6**: Although the intention is to be compatible with as may Python versions as possible, currently available code has only been tested with Python 3.11.6.
- **Graphviz** (optional): In order to plot the disjunctive graph of a `JobShopInstance` with the default layout you will need to have [PyGraphviz](https://pygraphviz.github.io/documentation/stable/install.html) and its dependencies.

## Project Structure
The project structure is based on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template. This template provides a standardized structure for organizing data science projects. Only minor modifications of this template has been made.

```python
|-- LICENSE
|-- Makefile            # Makefile with commands.
|-- README.md
|-- data
|   |-- README.md
|   |-- instances.json  # Json file with metadata of each instance.
|   `-- raw             # Orginal JSSP instances in Taillard form.
|       |-- abz5        # These files have no extension.
|       | ...
|       `-- yn4
|-- experiments         # Experiments' scripts.
|-- gnn_scheduler       # Source code for use in this project.
|   |-- __init__.py
|   |-- alphazero       # Future work.
|   |-- gnns            # Graph Neural Networks related source code.
|   |-- jssp            # Job Shop Scheduling related source code.
|-- notebooks           # Jupyter notebooks.
|   |-- data_exploration
|   `-- tutorial
|-- poetry.lock         # Dependency versions lock file.
|-- pyproject.toml      # Project and dependency configuration.
`-- tests               # Test scripts and unit tests for the project.
```


