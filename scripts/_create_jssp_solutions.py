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
    seed: int = 0,
    duration_range: tuple[int, int] = (1, 99),
):
    generator = GeneralInstanceGenerator(
        num_jobs=num_jobs,
        num_machines=num_machines,
        allow_less_jobs_than_machines=False,
        seed=seed,
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
        with open(
            get_data_path() / "raw" / file_name, "w", encoding="utf-8"
        ) as f:
            json.dump(schedules, f, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    # Train instances
    # _main(100_000)

    # Evaluation instances
    # _main(
    #     1_000,
    #     "instances10x10_eval_0.json",
    #     num_jobs=(10, 10),
    #     num_machines=(10, 10),
    # )
    # _main(
    #     1_000,
    #     "instances5x5_eval_0.json",
    #     num_jobs=(5, 5),
    #     num_machines=(5, 5),
    # )

    # _main(
    #     25_000,
    #     "instances10x10_train_1.json",
    #     num_jobs=(10, 10),
    #     num_machines=(10, 10),
    #     seed=1,
    # )
    # _main(
    #     50_000,
    #     "instances10x5_train_2.json",
    #     num_jobs=10,
    #     num_machines=5,
    #     seed=2,
    # )
    # _main(
    #     25_000,
    #     "instances10x10_train_3.json",
    #     num_jobs=10,
    #     num_machines=10,
    #     seed=3,
    # )
    # _main(
    #     50_000,
    #     "instances10x5_train_4.json",
    #     num_jobs=10,
    #     num_machines=5,
    #     seed=4,
    # )
    # _main(
    #     50_000,
    #     "instances10x5_train_5.json",
    #     num_jobs=10,
    #     num_machines=5,
    #     seed=5,
    #     duration_range=(50, 99),
    # )
    # _main(
    #     25_000,
    #     "instances10x5_train_6.json",
    #     num_jobs=10,
    #     num_machines=5,
    #     seed=6,
    #     duration_range=(50, 99),
    # )
    # _main(
    #     25_000,
    #     "instances10x5to10_train_7.json",
    #     num_jobs=10,
    #     num_machines=(5, 10),
    #     seed=7,
    #     duration_range=(50, 99),
    # )
    # _main(
    #     25_000,
    #     "instances10x5to10_train_8.json",
    #     num_jobs=10,
    #     num_machines=(5, 10),
    #     seed=8,
    # )
    # _main(
    #     25_000,
    #     "instances10x5to10_train_9.json",
    #     num_jobs=10,
    #     num_machines=(5, 10),
    #     seed=9,
    #     duration_range=(50, 99),
    # )
    # _main(
    #     25_000,
    #     "instances10x5to10_train_10.json",
    #     num_jobs=10,
    #     num_machines=(5, 10),
    #     seed=10,
    # )
    # _main(
    #     100,
    #     "instances5x5_testing_11.json",
    #     num_jobs=10,
    #     num_machines=5,
    #     seed=11,
    #     duration_range=(50, 99),
    # )
    # _main(
    #     100,
    #     "instances5x5_testing_12.json",
    #     num_jobs=10,
    #     num_machines=5,
    #     seed=12,
    # )
    # _main(
    #     100,
    #     "instances5x5_testing_13.json",
    #     num_jobs=10,
    #     num_machines=5,
    #     seed=13,
    #     duration_range=(50, 99),
    # )
    # _main(
    #     100,
    #     "instances5x5_testing_14.json",
    #     num_jobs=10,
    #     num_machines=5,
    #     seed=14,
    # )
    # for seed in range(15, 25):
    #    _main(
    #        10_000,
    #        f"instances8x8_train_{seed}.json",
    #        num_jobs=8,
    #        num_machines=8,
    #        seed=seed,
    #        duration_range=(50, 99),
    #    )
    for seed in range(25, 30):
        _main(
            1_000,
            f"instances10to15x5to10_train_{seed}.json",
            num_jobs=(10, 15),
            num_machines=(5, 10),
            seed=seed,
            duration_range=(50, 99),
        )
