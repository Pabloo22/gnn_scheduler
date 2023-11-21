from __future__ import annotations

from typing import Iterable
import os
import json
import pandas as pd

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
        # Lawrence 10x5 instance (Table 3, instance 2); also called (setf2) or (F2)
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

    raise NotImplementedError(f"Specification '{specification}' is not implemented.")


def load_from_benchmark(
    path: os.PathLike | str | bytes,
    instance_name: str,
    encoding: str = "utf-8",
    json_file: str = "instances.json",
) -> JobShopInstance:
    """Loads a job-shop instance from a benchmark file."""

    # get metadata from instances.json file
    instances_path = os.path.join(path, json_file)
    with open(instances_path, "r", encoding=encoding) as f:
        instances: list[dict] = json.load(f)

    optimum = None
    upper_bound = None
    lower_bound = None
    file_path = None
    for instance in instances:
        if instance["name"] != instance_name:
            continue
        optimum = instance["optimum"]
        if optimum is None and instance.get("bounds", None) is not None:
            upper_bound, lower_bound = instance["bounds"].values()
        file_path = os.path.join(path, instance["path"])
        break

    return JobShopInstance.load_from_file(
        file_path,
        name=instance_name,
        optimum=optimum,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        encoding=encoding,
    )


def load_metadata(
    path: os.PathLike | str | bytes,
    encoding: str = "utf-8",
    json_file: str = "instances.json",
) -> pd.DataFrame:
    """Loads the metadata from a benchmark file."""

    # get metadata from instances.json file
    instances_path = os.path.join(path, json_file)
    with open(instances_path, "r", encoding=encoding) as f:
        instances: list[dict] = json.load(f)

    return pd.DataFrame(instances)
