from __future__ import annotations

import itertools
from typing import NamedTuple, Optional, Callable

import networkx as nx


Layout = Callable[[nx.Graph], dict[str, tuple[float, float]]]


class Operation(NamedTuple):
    """Stores information about an operation in a job-shop scheduling
    problem."""

    machine_id: int
    duration: float

    def get_id(self, job_id: int, position: int) -> str:
        """Returns the id of the operation."""
        return f"J{job_id}M{self.machine_id}P{position}"


class JobShopInstance:
    """Stores a job-shop scheduling problem instance."""

    def __init__(
        self,
        jobs: list[list[Operation]],
        name: str = "JobShopInstance",
        optimum: Optional[float] = None,
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
    ):
        self.jobs = jobs
        self.n_machines = self._get_n_machines()
        self.name = name
        self.time = 0

        # List of lists of job ids. Each list represents a machine:
        self.current_solution = [[] for _ in range(self.n_machines)]

        self.optimum = optimum
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self._disjunctive_graph = None

    @property
    def n_jobs(self) -> int:
        """Returns the number of jobs in the instance."""
        return len(self.jobs)

    @property
    def disjunctive_graph(self) -> nx.DiGraph:
        """Returns the disjunctive graph of the instance."""
        if self._disjunctive_graph is None:
            self._disjunctive_graph = self._create_disjunctive_graph()
        return self._disjunctive_graph

    @property
    def bounds(self) -> tuple[float, float]:
        """Returns the lower and upper bounds of the instance."""
        return self.lower_bound, self.upper_bound

    @property
    def total_duration(self) -> float:
        """Returns the total duration of the instance."""
        total_duration = 0
        for job in self.jobs:
            total_duration += sum(operation.duration for operation in job)
        return total_duration

    def _get_n_machines(self) -> int:
        """Returns the number of machines in the instance."""
        mx = 0
        for job in self.jobs:
            mx_machine = max(operation.machine_id for operation in job)
            mx = max(mx, mx_machine)
        return mx + 1