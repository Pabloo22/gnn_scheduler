from typing import NamedTuple

import attrs


class Operation(NamedTuple):
    """A class representing an operation in a job-shop scheduling problem."""

    job: int
    machine: int
    duration: float


@attrs.define
class JobShopInstance:
    """A class representing a job-shop scheduling problem instance."""

    n_jobs: int
    n_machines: int
    operations: list[Operation] = attrs.Factory(list)
