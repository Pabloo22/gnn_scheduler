from __future__ import annotations

import random

import tqdm

from gnn_scheduler import get_data_path
from gnn_scheduler.jssp import (
    JobShopInstance,
    load_all_from_benchmark,
    load_pickle_instances,
    RemoveJobs,
    RemoveMachines,
    AddDurationNoise,
)
from gnn_scheduler.jssp import set_instance_attributes



def augment_instances(
    instances: list[JobShopInstance], show_progress: bool = True
) -> list[JobShopInstance]:
    """Augments a list of JobShopInstance objects through a three-step process.

    1. Generate instances with exactly 10 machines.
    2. For each of these new instances, generate instances with 10 to 20 jobs.
    3. For each new instance, add noise to the durations.

    Args:
        instances: List of original JobShopInstance objects.
        show_progress: Whether to show a progress bar.

    Returns:
        A list of new augmented JobShopInstance objects.
    """
    random.seed(0)

    # Step 1: Reduce to 10 machines
    new_instances_step1 = []
    remove_machines = RemoveMachines(n_machines=10)
    for instance in tqdm.tqdm(
        instances, disable=not show_progress, desc="Reducing to 10 machines"
    ):
        if instance.n_machines > 10:
            for _ in range(instance.n_machines - 10):
                new_instance = remove_machines(instance)
                new_instances_step1.append(new_instance)
        elif instance.n_machines == 10:
            new_instances_step1.append(instance)

    # Step 2: Adjust the number of jobs for each instance from Step 1
    new_instances_step2 = []
    remove_jobs = RemoveJobs(min_jobs=10, max_jobs=20)
    for instance in tqdm.tqdm(
        new_instances_step1, disable=not show_progress, desc="Adjusting jobs"
    ):
        if instance.n_jobs > 10:
            for _ in range(instance.n_jobs - 10):
                new_instance_with_less_jobs = remove_jobs(instance)
                new_instances_step2.append(new_instance_with_less_jobs)

    # Step 3: Add noise to the durations for each instance from Step 2
    new_instances_step3 = []
    add_noise = AddDurationNoise()
    for instance in tqdm.tqdm(
        new_instances_step2, disable=not show_progress, desc="Adding noise"
    ):
        new_instance_with_noise = add_noise(instance)
        new_instances_step3.append(new_instance_with_noise)

    return new_instances_step3


def augment_benchmark(folder_name: str = "augmented_benchmark_10machines"):
    """Augments the benchmark and saves the new instances to disk."""
    instances = load_all_from_benchmark()
    new_instances = augment_instances(instances)

    path = get_data_path() / folder_name
    for instance in new_instances:
        instance.save(path / (instance.name + ".pkl"))


def label_all_instances(
    folder_name: str = "augmented_benchmark_10machines",
    show_progress: bool = True,
):
    """Labels all instances in the augmented benchmark."""
    instances = load_pickle_instances(folder_name, show_progress=show_progress)
    for instance in tqdm.tqdm(
        instances,
        disable=not show_progress,
        desc="Setting instance attributes",
    ):
        set_instance_attributes(instance, time_limit=0.1)

    path = get_data_path() / folder_name
    for instance in tqdm.tqdm(
        instances, disable=not show_progress, desc="Saving instances"
    ):
        instance.save(path / (instance.name + ".pkl"))


if __name__ == "__main__":
    # augment_benchmark()
    label_all_instances()
