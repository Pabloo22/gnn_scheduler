import random
from typing import Optional

from gnn_scheduler.job_shop import JobShopInstance, Operation


class NaiveGenerator:
    def __init__(
        self,
        max_duration: int,
        min_duration: int = 1,
        max_n_jobs: int = 20,
        min_n_jobs: int = 2,
        max_n_machines: int = 10,
        name_suffix: str = "naive_generated_instance",
        seed: Optional[int] = None,
    ):
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.min_n_jobs = min_n_jobs
        self.name_suffix = name_suffix
        self.counter = 0
        self.seed = seed
        random.seed(self.seed)

    def generate(
        self, n_jobs: Optional[int] = None, n_machines: Optional[int] = None
    ) -> JobShopInstance:
        n_jobs, n_machines = self._get_n_jobs_and_n_machines(
            n_jobs, n_machines
        )
        jobs = []
        available_machines = list(range(n_machines))
        for _ in range(n_jobs):
            job = []
            for _ in range(n_machines):
                operation = self._create_random_operation(available_machines)
                job.append(operation)
            jobs.append(job)
            available_machines = list(range(n_machines))

        return JobShopInstance(jobs=jobs, name=self._get_name())

    def _create_random_operation(
        self, available_machines: list[int]
    ) -> Operation:
        machine_id = random.choice(available_machines)
        available_machines.remove(machine_id)
        duration = random.randint(self.min_duration, self.max_duration)
        return Operation(machine_id=machine_id, duration=duration)

    def _get_n_jobs_and_n_machines(
        self, n_jobs: int, n_machines: int
    ) -> tuple[int, int]:
        if n_jobs is None:
            n_jobs = random.randint(self.min_n_jobs, self.max_n_jobs)
        if n_machines is None:
            n_machines = random.randint(n_jobs, self.max_n_machines)
        return n_jobs, n_machines

    def _get_name(self) -> str:
        self.counter += 1
        return f"{self.name_suffix}_{self.counter}"
