from __future__ import annotations

import os
from typing import Optional
import tqdm

from gnn_scheduler import get_project_path
from gnn_scheduler.jssp import(
    NaiveGenerator,
    CPSolver,
)


def create_dataset(folder_name: str = "diff_prediction_instances",
                   n_instances: int = 50_000,
                   min_jobs: int = 10,
                   max_jobs: int = 20,
                   machines: int = 10,
                   min_duration: int = 1,
                   max_duration: int = 100,
                   time_limit: float = 0.1,
                   seed: int = 0,
                   path: Optional[os.PathLike | str | bytes] = None,
                   show_progress: bool = True,
                   ) -> None:
    """Creates a dataset of job-shop instances."""

    generator = NaiveGenerator(
        max_n_jobs=max_jobs,
        max_duration=max_duration,
        min_n_jobs=min_jobs,
        min_duration=min_duration,
        seed=seed,
    )

    # Add folder to path
    path = get_project_path() if path is None else path
    data_path = os.path.join(path, "data")
    path_with_folder = os.path.join(data_path, folder_name)

    for _ in tqdm.tqdm(range(n_instances), disable=not show_progress):
        instance = generator.generate(n_machines=machines)
        solver = CPSolver(instance, time_limit=time_limit)
        solution = solver.solve()
        instance.lower_bound = instance.max_machine_load

        if solution is not None:
            instance.upper_bound = solution["makespan"]
            if solution["status"] == "optimal":
                instance.optimum = solution["makespan"]
        else:
            instance.upper_bound = instance.max_machine_load * 2

        instance_path = os.path.join(path_with_folder, instance.name + ".pkl")
        instance.save(instance_path)


if __name__ == "__main__":
    create_dataset()
