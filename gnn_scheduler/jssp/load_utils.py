from __future__ import annotations

from typing import Iterable, Optional
import os
import json

import tqdm

from gnn_scheduler import get_data_path
from gnn_scheduler.jssp import JobShopInstance, Operation


def _read_taillard_file(
    lines: Iterable[str],
    comment_symbol: str = "#",
    **kwargs,
) -> JobShopInstance:
    """Returns a job-shop instance from a Taillard file.

    Example of a Taillard file:
        #+++++++++++++++++++++++++++++
        # instance la02
        #+++++++++++++++++++++++++++++
        # Lawrence 10x5 instance (Table 3, instance 2); also called...
        10 5
        0 20 3 87 1 31 4 76 2 17
        4 25 2 32 0 24 1 18 3 81
        1 72 2 23 4 28 0 58 3 99
        2 86 1 76 4 97 0 45 3 90
        4 27 0 42 3 48 2 17 1 46
        1 67 0 98 4 48 3 27 2 62
        4 28 1 12 3 19 0 80 2 50
        1 63 0 94 2 98 3 50 4 80
        4 14 0 75 2 50 1 41 3 55
        4 72 2 18 1 37 3 79 0 61
    """

    first_non_comment_line_reached = False
    jobs = []
    for line in lines:
        line = line.strip()
        if line.startswith(comment_symbol):
            continue
        if not first_non_comment_line_reached:
            first_non_comment_line_reached = True
            continue

        row = list(map(int, line.split()))

        pairs = zip(row[::2], row[1::2])
        operations = [
            Operation(machine_id=machine_id, duration=duration)
            for machine_id, duration in pairs
        ]
        jobs.append(operations)

    return JobShopInstance(jobs=jobs, **kwargs)


def load_from_file(
    path: os.PathLike | str | bytes,
    comment_symbol: str = "#",
    specification: str = "taillard",
    encoding: str = "utf-8",
    **kwargs,
) -> JobShopInstance:
    """Loads a job-shop instance from a file."""

    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()

    if specification == "taillard":
        return _read_taillard_file(lines, comment_symbol, **kwargs)

    raise NotImplementedError(
        f"Specification '{specification}' is not implemented."
    )


def load_metadata(
    path: Optional[os.PathLike | str | bytes] = None,
    encoding: str = "utf-8",
    json_file: str = "instances.json",
    if_has_optimum: bool = False,
    list_of_instances: Optional[list[str]] = None,
    max_jobs: Optional[int] = None,
    max_machines: Optional[int] = None,
) -> Iterable[dict]:
    """Loads the metadata from a benchmark file."""

    if path is None:
        path = get_data_path()

    # get metadata from instances.json file
    metadata_path = os.path.join(path, json_file)
    with open(metadata_path, "r", encoding=encoding) as f:
        metadata: list[dict] = json.load(f)

    if if_has_optimum:
        metadata = [
            instance
            for instance in metadata
            if instance["optimum"] is not None
        ]
    if list_of_instances is not None:
        metadata = [
            instance
            for instance in metadata
            if instance["name"] in list_of_instances
        ]
    if max_jobs is not None:
        metadata = [
            instance for instance in metadata if instance["jobs"] <= max_jobs
        ]
    if max_machines is not None:
        metadata = [
            instance
            for instance in metadata
            if instance["machines"] <= max_machines
        ]

    return metadata


def load_from_benchmark(
    instance_name: str,
    path: Optional[os.PathLike | str | bytes] = None,
    encoding: str = "utf-8",
    json_file: str = "instances.json",
    metadata: Optional[list[dict]] = None,
) -> JobShopInstance:
    """Loads a job-shop instance from a benchmark file."""

    if path is None:
        path = get_data_path()

    if metadata is None:
        metadata = load_metadata(path, encoding, json_file)

    optimum = None
    upper_bound = None
    lower_bound = None
    file_path = None
    for instance in metadata:
        if instance["name"] != instance_name:
            continue
        optimum = instance["optimum"]
        if "bounds" in instance:
            upper_bound, lower_bound = instance["bounds"].values()
        else:
            upper_bound = optimum
            lower_bound = optimum
        file_path = os.path.join(path, instance["path"])
        break

    return load_from_file(
        file_path,
        name=instance_name,
        optimum=optimum,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        encoding=encoding,
    )


def load_all_from_benchmark(
    path: Optional[os.PathLike | str | bytes] = None,
    encoding: str = "utf-8",
    json_file: str = "instances.json",
    max_jobs: Optional[int] = None,
    max_machines: Optional[int] = None,
    list_of_instances: Optional[list[str]] = None,
    if_has_optimum: bool = False,
    metadata: Optional[list[dict]] = None,
) -> list[JobShopInstance]:
    """Loads all job-shop instances."""

    if path is None:
        path = get_data_path()
    if metadata is None:
        metadata = load_metadata(
            path=path,
            encoding=encoding,
            json_file=json_file,
            if_has_optimum=if_has_optimum,
            list_of_instances=list_of_instances,
            max_jobs=max_jobs,
            max_machines=max_machines,
        )

    instances = []
    for instance in metadata:
        instance_name = instance["name"]
        instance = load_from_benchmark(
            instance_name, path, encoding, json_file, metadata
        )
        instances.append(instance)
    return instances


def load_pickle_instances(
    folder_name: str,
    data_path: Optional[os.PathLike | str | bytes] = None,
    show_progress: bool = True,
):
    """Loads all instances from a folder containing pickle files."""

    if data_path is None:
        data_path = get_data_path()

    instances = []
    for file_name in tqdm.tqdm(
        os.listdir(data_path / folder_name),
        disable=not show_progress,
        desc="Loading instances",
    ):
        if file_name.endswith(".pkl"):
            instance = JobShopInstance.load(
                data_path / folder_name / file_name
            )
            instances.append(instance)
    return instances


def load_pickle_instances_from_folders(
    folder_names: list[str],
    show_progress: bool = True,
    data_path: Optional[os.PathLike | str | bytes] = None,
) -> list[JobShopInstance]:
    """Loads all instances from the given folders."""
    instances = []
    for folder_name in folder_names:
        instances.extend(
            load_pickle_instances(
                folder_name, show_progress=show_progress, data_path=data_path
            )
        )
    return instances
