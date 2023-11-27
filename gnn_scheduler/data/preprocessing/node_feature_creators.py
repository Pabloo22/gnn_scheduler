from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx

from gnn_scheduler.data.preprocessing import (get_n_jobs, 
                                              get_n_machines,
                                              get_job_loads,
                                              )


class NodeFeatureCreator(ABC):
    """Base class for node feature creators."""

    def __init__(self):
        self.graph = None
        self.is_fit = False
        self.name = self.__class__.__name__

    def fit(self, graph: nx.DiGraph):
        """Used to fit the feature creator to the graph.

        Stores the graph.

        It can be implemented by sub-classes if necessary to store more
        information.
        """
        self.graph = graph
        self.is_fit = True

    @abstractmethod
    def create_features(self, node_name: str, node_data: dict[str, Any]) -> list[float]:
        """_summary_

        Args:
            node_name (str): name of the node
            node_data (dict[str, Any]): data associated with the node

        Returns:
            list[float]: a list of floats representing the features of the node
        """

    def __str__(self) -> str:
        return f"{self.name}()"

    def __call__(self, node_name: str, node_data: dict[str, Any]) -> list[float]:
        if not self.is_fit:
            raise RuntimeError(f"{self.name} is not fit.")

        return self.create_features(node_name, node_data)


class InAndOutDegrees(NodeFeatureCreator):
    """The normalized in- and out-degrees of a node."""

    def fit(self, graph: nx.DiGraph):
        """Stores the graph"""
        self.graph = graph

    def create_node_features(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        """Returns the normalized in- and out-degrees of a node.

        Args:
            node_name (str): name of the node
            node_data (dict[str, Any]): data associated with the node
            graph (nx.DiGraph): the networkx graph

        Returns:
            list[float]:
        """
        in_degree = self.graph.in_degree(node_name) / (self.graph.number_of_nodes() - 1)
        out_degree = self.graph.out_degree(node_name) / (
            self.graph.number_of_nodes() - 1
        )
        return [in_degree, out_degree]


class OneHotEncoding(NodeFeatureCreator):
    """One-hot encoding of a node attribute.

    Args:
        NodeFeatureCreator (_type_):
    """

    def __init__(self, feature_name: str, n_values: int = 100):
        super().__init__()
        self.feature_name = feature_name
        self.n_values = n_values

    def create_node_fetures(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        """Creates the one-hot encoding of the node attribute.

        Args:
            node_name (str):
            node_data (dict[str, Any]):
            graph (nx.DiGraph):

        Returns:
            list[float]:
        """
        zeros = [0.0] * self.n_values
        machine_id = node_data.get(self.feature_name)
        if machine_id is not None and 0 <= machine_id < self.n_values:
            zeros[machine_id] = 1.0
        return zeros


class Duration(NodeFeatureCreator):
    """The processing time required for each operation.

    It is normalized by the maximum operation time across the graph to ensure
    this feature falls within a consistent range."""

    def __init__(self):
        super().__init__()
        self.max_duration = 0.0

    def fit(self, graph: nx.DiGraph):
        """Calculates the maximum operation time across the graph."""
        for node in graph.nodes:
            self.max_duration = max(self.max_duration, graph.nodes[node]["duration"])
        self.is_fit = True

    def create_features(self, node_name: str, node_data: dict[str, Any]) -> list[float]:
        return [node_data["duration"] / self.max_duration]


class MachineLoad(NodeFeatureCreator):
    """How utilized each machine is.

    A calculation of the total processing time that is scheduled for each
    machine, normalized by the maximum load across all machines, to provide a
    sense of how utilized each machine is.
    """

    def __init__(self):
        super().__init__()
        self.max_load = 0
        self.machines_load = None

    def fit(self, graph: nx.DiGraph):
        """Calculates the maximum load across all machines."""
        self.graph = graph
        machines_load = [0] * get_n_machines(graph)
        for _, node_data in graph.nodes(data=True):
            machines_load[node_data["machine_id"]] += node_data["duration"]

        self.machines_load = machines_load
        self.max_load = max(machines_load)
        self.is_fit = True

    def create_features(self, node_name: str, node_data: dict[str, Any]) -> list[float]:
        machine_id = node_data["machine_id"]
        return [self.machines_load[machine_id] / self.max_load]


class JobLoad(NodeFeatureCreator):
    """The cumulative processing time of each job."""

    def __init__(self):
        super().__init__()
        self.job_loads = None
        self.max_load = 0

    def fit(self, graph: nx.DiGraph):
        """Calculates the maximum load across all jobs."""
        self.graph = graph
        job_loads = get_job_loads(graph)
        self.max_load = max(job_loads.values())
        self.is_fit = True

    def create_features(self, node_name: str, node_data: dict[str, Any]) -> list[float]:
        job_id = node_data["job_id"]
        return [self.job_loads[job_id] / self.max_load]


class OperationIndex(NodeFeatureCreator):
    """The index of the operation in the job.

    The sequential position of an operation within its job, represented as a
    fraction of the total number of operations in the job, to give the model
    insight into the operation's order without depending on the absolute
    number of operations.
    """

    def __init__(self):
        super().__init__()
        self.n_operations_per_job = None

    def fit(self, graph: nx.DiGraph):
        """Calculates the number of operations per job."""
        self.graph = graph
        n_operations_per_job = [0] * get_n_jobs(graph)
        for _, node_data in graph.nodes(data=True):
            n_operations_per_job[node_data["job_id"]] += 1
        
        self.n_operations_per_job = n_operations_per_job
        self.is_fit = True
    
    def create_features(self, node_name: str, node_data: dict[str, Any]) -> list[float]:
        job_id = node_data["job_id"]
        position = node_data["position"] + 1
        return [position / self.n_operations_per_job[job_id]]
    
    


if __name__ == "__main__":
    pass
