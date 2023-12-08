from __future__ import annotations

import pandas as pd

from gnn_scheduler.jssp import JobShopInstance, CPSolver


def set_instance_attributes(
    instance: JobShopInstance, time_limit: float = 0.1
) -> JobShopInstance:
    """Sets the lower and upper bounds of the instance."""

    solver = CPSolver(instance, time_limit=time_limit)
    solution = solver.solve()
    instance.lower_bound = instance.max_machine_load

    if solution is not None:
        instance.upper_bound = solution["makespan"]
        if solution["status"] == "optimal":
            instance.optimum = solution["makespan"]
    else:
        instance.upper_bound = instance.max_machine_load * 2

    return instance


def get_difficulty_score(instance: JobShopInstance) -> float:
    """Returns the difficulty score of an instance."""

    if instance.upper_bound is None:
        return 1
    return instance.upper_bound / instance.lower_bound - 1


def get_stat_dataframe(instances: list[JobShopInstance]):
    names = []
    lower_bounds = []
    upper_bounds = []
    is_optimal = []
    n_jobs = []
    n_machines = []
    no_solutions = []  # True if upper_bound is None
    # upper_bound / max(machine_load, max_job_duration) - 1:
    difficulty_scores = []

    for instance in instances:
        names.append(instance.name)

        max_machine_load_and_job_duration = max(
            instance.max_machine_load, instance.max_job_duration
        )
        instance.lower_bound = max_machine_load_and_job_duration
        lower_bounds.append(max_machine_load_and_job_duration)
        n_jobs.append(instance.n_jobs)
        n_machines.append(instance.n_machines)
        is_optimal.append(instance.optimum is not None)

        if instance.upper_bound is None:
            instance.upper_bound = instance.lower_bound * 2
        no_solution = instance.upper_bound == instance.lower_bound * 2
        no_solutions.append(no_solution)
        upper_bounds.append(instance.upper_bound)

        difficulty_scores.append(
            instance.upper_bound / instance.lower_bound - 1
        )

    # Create dataframe
    df = pd.DataFrame(
        {
            "name": names,
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            "max_machine_load_and_job_duration": lower_bounds,
            "upper_bound": upper_bounds,
            "is_optimal": is_optimal,
            "no_solution": no_solutions,
            "difficulty_score": difficulty_scores,
        }
    )

    return df
