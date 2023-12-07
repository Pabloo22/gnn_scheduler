from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gnn_scheduler.jssp.graphs import DisjunctiveGraph


class NodeFeatureCreator(ABC):
    """Base class for node feature creators."""

    def __init__(self):
        self.graph = None
        self.is_fit = False
        self.name = self.__class__.__name__

    def fit(self, graph: DisjunctiveGraph):
        """Used to fit the feature creator to the graph.

        Stores the graph.

        It can be implemented by sub-classes if necessary to store more
        information.
        """
        self.graph = graph
        self.is_fit = True

    @abstractmethod
    def create_features(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        """Creates the features of a node.

        This method should be implemented by sub-classes.

        Args:
            node_name (str): name of the node
            node_data (dict[str, Any]): data associated with the node

        Returns:
            list[float]: a list of floats representing the features of the node
        """

    def __str__(self) -> str:
        return f"{self.name}()"

    def __call__(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        if not self.is_fit:
            raise RuntimeError(f"{self.name} is not fit.")

        return self.create_features(node_name, node_data)


class InAndOutDegrees(NodeFeatureCreator):
    """The normalized in- and out-degrees of a node."""

    def create_node_features(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        """Returns the normalized in- and out-degrees of a node.

        Args:
            node_name (str): name of the node
            node_data (dict[str, Any]): data associated with the node
            graph (DisjunctiveGraph): the networkx graph

        Returns:
            list[float]:
        """
        in_degree = self.graph.in_degree(node_name) / (
            self.graph.number_of_nodes() - 1
        )
        out_degree = self.graph.out_degree(node_name) / (
            self.graph.number_of_nodes() - 1
        )
        return [in_degree, out_degree]


class OneHotEncoding(NodeFeatureCreator):
    """One-hot encoding of a node attribute."""

    def __init__(self, feature_name: str, n_values: int):
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
            graph (DisjunctiveGraph):

        Returns:
            list[float]:
        """
        zeros = [0.0] * self.n_values
        feature_id = node_data.get(self.feature_name)
        if feature_id is not None and 0 <= feature_id < self.n_values:
            zeros[feature_id] = 1.0
        return zeros


class Duration(NodeFeatureCreator):
    """The processing time required for each operation.

    It is normalized by the maximum operation time across the graph/job/machine
    to ensure this feature falls within a consistent range."""

    def __init__(
        self,
        normalize_with: str = "graph",
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__()
        self.normalize_with = normalize_with
        self.min_value = min_value
        self.max_value = max_value

    def create_features(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        if self.normalize_with == "graph":
            max_duration = self.graph.max_graph_duration
        elif self.normalize_with == "machine":
            machine_id = node_data["machine_id"]
            max_duration = self.graph.max_machine_durations[machine_id]
        elif self.normalize_with == "job":
            job_id = node_data["job_id"]
            max_duration = self.graph.max_job_durations[job_id]
        else:
            raise ValueError(
                f"Unknown normalization option: {self.normalize_with}"
            )
        percentage = node_data["duration"] / max_duration
        return [
            self.min_value + percentage * (self.max_value - self.min_value)
        ]


class MachineLoad(NodeFeatureCreator):
    """How utilized each machine is.

    A calculation of the total processing time that is scheduled for each
    machine, normalized by the maximum load across all machines, to provide a
    sense of how utilized each machine is.
    """

    def create_features(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        machine_id = node_data["machine_id"]
        machine_load = self.graph.machines_load[machine_id]
        max_load = self.graph.max_machine_load
        return [machine_load / max_load]


class JobLoad(NodeFeatureCreator):
    """The cumulative processing time of each job."""

    def __init__(self):
        super().__init__()
        self.job_loads = None
        self.max_load = 0

    def create_features(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        job_id = node_data["job_id"]
        job_load = self.graph.job_loads[job_id]
        max_load = self.graph.max_job_load
        return [job_load / max_load]


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

    def create_features(
        self, node_name: str, node_data: dict[str, Any]
    ) -> list[float]:
        job_id = node_data["job_id"]
        position = node_data["position"] + 1
        n_operations = self.graph.n_operations_per_job[job_id]
        return [position / n_operations]
