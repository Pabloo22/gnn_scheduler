import json
import tqdm
from job_shop_lib.generation import GeneralInstanceGenerator
from job_shop_lib.constraint_programming import ORToolsSolver
from gnn_scheduler.utils import get_data_path


def _main(n_instances: int, file_name: str = "small_random_instances_0.json"):
    generator = GeneralInstanceGenerator(
        num_jobs=(3, 6),
        num_machines=(3, 5),
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
    _main(100_000)
