from __future__ import annotations

import abc
import copy
import random
from typing import Optional

from gnn_scheduler.jssp import JobShopInstance, Operation


class Transformation(abc.ABC):
    """Base class for transformations applied to JobShopInstance objects."""

    @abc.abstractmethod
    def apply(self, instance: JobShopInstance) -> JobShopInstance:
        """Applies the transformation to a given JobShopInstance.

        Args:
            instance: The JobShopInstance to transform.

        Returns:
            A new JobShopInstance with the transformation applied.
        """

    def __call__(self, instance: JobShopInstance) -> JobShopInstance:
        return self.apply(instance)


class RemoveMachines(Transformation):
    """Removes operations associated with randomly selected machines until
    there are exactly n_machines machines left."""

    def __init__(self, n_machines: int):
        self.n_machines = n_machines

    @staticmethod
    def remove_machine(
        instance: JobShopInstance, machine_id: int
    ) -> JobShopInstance:
        machine_id = random.choice(range(instance.n_machines))
        new_jobs = []
        for job in instance.jobs:
            new_jobs.append([op for op in job if op.machine_id != machine_id])

        # Adjust the machine indices
        for job in new_jobs:
            for op in job:
                if op.machine_id > machine_id:
                    op.machine_id -= 1

        return JobShopInstance(new_jobs, instance.name)

    def apply(self, instance: JobShopInstance) -> JobShopInstance:
        while instance.n_machines > self.n_machines:
            instance = RemoveMachines.remove_machine(
                instance, random.choice(range(instance.n_machines))
            )

        return instance


class AddDurationNoise(Transformation):
    """Adds uniform integer noise to operation durations."""

    def __init__(
        self,
        min_duration: float = 1.0,
        max_duration: float = 100.0,
        noise_level: int = 10,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.noise_level = noise_level

    def apply(self, instance: JobShopInstance) -> JobShopInstance:
        new_jobs = []
        for job in instance.jobs:
            new_job = []
            for op in job:
                noise = random.randint(-self.noise_level, self.noise_level)
                new_duration = max(
                    self.min_duration,
                    min(self.max_duration, op.duration + noise),
                )

                new_job.append(Operation(op.machine_id, new_duration))
            new_jobs.append(new_job)

        return JobShopInstance(new_jobs, instance.name)


class RemoveJobs(Transformation):
    """Removes jobs randomly until the number of jobs is within a specified
    range."""

    def __init__(
        self, min_jobs: int, max_jobs: int, target_jobs: Optional[int] = None
    ):
        """
        Args:
            min_jobs: The minimum number of jobs to remain in the instance.
            max_jobs: The maximum number of jobs to remain in the instance.
            target_jobs: If specified, the number of jobs to remain in the
                instance. Overrides min_jobs and max_jobs.
        """
        self.min_jobs = min_jobs
        self.max_jobs = max_jobs
        self.target_jobs = target_jobs

    def apply(self, instance: JobShopInstance) -> JobShopInstance:
        if self.target_jobs is None:
            target_jobs = random.randint(self.min_jobs, self.max_jobs)
        else:
            target_jobs = self.target_jobs
        new_jobs = copy.deepcopy(instance.jobs)

        while len(new_jobs) > target_jobs:
            new_jobs.pop(random.randint(0, len(new_jobs) - 1))

        return JobShopInstance(new_jobs, instance.name)

    @staticmethod
    def remove_job(
        instance: JobShopInstance, job_index: int
    ) -> JobShopInstance:
        """Removes a specific job from the instance.

        Args:
            instance: The JobShopInstance from which to remove the job.
            job_index: The index of the job to remove.

        Returns:
            A new JobShopInstance with the specified job removed.
        """
        new_jobs = copy.deepcopy(instance.jobs)
        new_jobs.pop(job_index)
        return JobShopInstance(new_jobs, instance.name)
