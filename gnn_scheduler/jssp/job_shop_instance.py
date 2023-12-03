from __future__ import annotations

import functools
from typing import NamedTuple, Optional


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
        self.name = name
        self.time = 0

        # List of lists of job ids. Each list represents a machine:
        self.current_solution = [[] for _ in range(self.n_machines)]

        self.optimum = optimum
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    @property
    def n_jobs(self) -> int:
        """Returns the number of jobs in the instance."""
        return len(self.jobs)

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

    @functools.cached_property
    def n_machines(self) -> int:
        """Returns the number of machines in the instance."""
        mx = 0
        for job in self.jobs:
            mx_machine = max(operation.machine_id for operation in job)
            mx = max(mx, mx_machine)
        return mx + 1

    @functools.cached_property
    def max_duration(self) -> float:
        """Returns the maximum duration of the instance."""
        mx = 0
        for job in self.jobs:
            mx_duration = max(operation.duration for operation in job)
            mx = max(mx, mx_duration)
        return mx

    @functools.cached_property
    def disjunctive_graph(self):
        """Returns the disjunctive graph of the instance."""
        # Imported here to avoid circular imports
        from gnn_scheduler.jssp.graphs import DisjunctiveGraph

        return DisjunctiveGraph.from_job_shop_instance(self)

    @functools.cached_property
    def max_job_duration(self) -> float:
        """Returns the maximum duration of a job in the instance."""
        return max(sum(operation.duration for operation in job) 
                   for job in self.jobs)

    @functools.cached_property
    def machine_loads(self) -> list[float]:
        """Returns the total duration of each machine in the instance."""
        machine_times = [0 for _ in range(self.n_machines)]
        for job in self.jobs:
            for operation in job:
                machine_times[operation.machine_id] += operation.duration
        return machine_times

    @functools.cached_property
    def max_machine_load(self) -> float:
        """Returns the maximum duration of a machine in the instance."""
        return max(self.machine_loads)
