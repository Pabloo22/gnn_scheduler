import json
import tqdm
from job_shop_lib.generation import GeneralInstanceGenerator
from job_shop_lib.constraint_programming import ORToolsSolver
from gnn_scheduler.utils import get_data_path


def _main(
    n_instances: int,
    file_name: str = "small_random_instances_0.json",
    num_jobs: int | tuple[int, int] = (3, 6),
    num_machines: int | tuple[int, int] = (3, 5),
):
    generator = GeneralInstanceGenerator(
        num_jobs=num_jobs,
        num_machines=num_machines,
        allow_less_jobs_than_machines=False,
        seed=0,
        name_suffix="small_random_instance",
        iteration_limit=n_instances,
    )
    schedules = []
    try:
        for instance in tqdm.tqdm(generator):
            solver = ORToolsSolver()
            schedule = solver(instance)
            schedules.append(schedule.to_dict())
    finally:
        with open(get_data_path() / file_name, "w", encoding="utf-8") as f:
            json.dump(schedules, f, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    # Train instances
    # _main(100_000)

    # Evaluation instances
    _main(
        1_000,
        "instances10x10_eval_0.json",
        num_jobs=(10, 10),
        num_machines=(10, 10),
    )
    _main(
        1_000,
        "instances5x5_eval_0.json",
        num_jobs=(5, 5),
        num_machines=(5, 5),
    )
