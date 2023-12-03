from __future__ import annotations

import random

from gnn_scheduler.jssp import JobShopInstance, Operation


class NaiveGenerator:
    def __init__(
        self,
        max_n_jobs: int,
        max_n_machines: int,
        max_duration: int,
        name_suffix: str = "naive_generated_instance",
    ):
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_duration = max_duration
        self.name_suffix = name_suffix
        self.counter = 0

    def generate(self) -> JobShopInstance:
        n_jobs = random.randint(1, self.max_n_jobs)
        n_machines = random.randint(1, self.max_n_machines)
        jobs = []
        available_machines = list(range(n_machines))
        for _ in range(n_jobs):
            operations = []
            for _ in range(n_machines):
                machine_id = random.choice(available_machines)
                # remove machine from available machines
                available_machines.remove(machine_id)
                duration = random.randint(1, self.max_duration)
                operations.append(
                    Operation(machine_id=machine_id, duration=duration)
                )
            jobs.append(operations)

        return JobShopInstance(jobs=jobs, name=self._get_name())

    def _get_name(self) -> str:
        self.counter += 1
        return f"{self.name_suffix}_{self.counter}"
