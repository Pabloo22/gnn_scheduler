"""Contains the JobShopInstance and Operation classes."""
from __future__ import annotations

import functools
from typing import Optional, Any, NamedTuple
import os
import pickle


class Operation(NamedTuple):
    machine_id: int
    duration: int

    def get_id(self, job_id: int, position: int) -> str:
        return f"J{job_id}M{self.machine_id}P{position}"

class JobShopInstance:
    """Stores a classic job-shop scheduling problem instance."""

    def __init__(
        self,
        jobs: list[list[Operation]],
        name: str = "JobShopInstance",
        **metadata: Any,
    ):
        self.jobs = jobs
        self.name = name
        self.metadata = metadata

    def to_dict(self):
        return {
            "jobs": self.jobs,
            "name": self.name,
            "metadata": self.metadata,
        }

    @property
    def n_jobs(self) -> int:
        """Returns the number of jobs in the instance."""
        return len(self.jobs)

    @property
    def bounds(self) -> tuple[float, float]:
        """Returns the lower and upper bounds of the instance."""
        return self.lower_bound, self.upper_bound

    @property
    def upper_bound(self) -> Optional[float]:
        """Returns the upper bound of the instance."""
        return self.metadata.get("upper_bound")

    @property
    def lower_bound(self) -> Optional[float]:
        """Returns the lower bound of the instance."""
        return self.metadata.get("lower_bound")

    @property
    def optimum(self) -> Optional[float]:
        """Returns the optimum of the instance."""
        return self.metadata.get("optimum")

    @functools.cached_property
    def disjunctive_graph(self):
        """Returns the disjunctive graph of the instance."""
        # Imported here to avoid circular imports
        from gnn_scheduler.job_shop.graphs import DisjunctiveGraph

        return DisjunctiveGraph.from_job_shop_instance(self)

    @functools.cached_property
    def job_durations(self) -> list[float]:
        """Returns the duration of each job in the instance."""
        return [
            sum(operation.duration for operation in job) for job in self.jobs
        ]

    @functools.cached_property
    def total_duration(self) -> float:
        """Returns the total duration of the instance."""
        return sum(self.job_durations)

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
    def max_job_duration(self) -> float:
        """Returns the maximum duration of a job in the instance."""
        return max(
            sum(operation.duration for operation in job) for job in self.jobs
        )

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

    @functools.cached_property
    def mean_machine_load(self) -> float:
        """Returns the mean duration of a machine in the instance."""
        return self.total_duration / self.n_machines

    def save(self, path: os.PathLike | str | bytes):
        """Uses pickle to save the instance to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: os.PathLike | str | bytes) -> JobShopInstance:
        """ "Uses pickle to load the instance from a file."""
        with open(path, "rb") as f:
            instance = pickle.load(f)
        return instance
