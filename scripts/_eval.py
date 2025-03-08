from collections.abc import Iterable, Callable
from functools import partial

import torch
import tqdm
from typing import List, Optional
from job_shop_lib import JobShopInstance, Schedule
from job_shop_lib.benchmarking import load_all_benchmark_instances
from job_shop_lib.dispatching.rules import (
    DispatchingRuleSolver,
    DispatchingRuleType,
)

from gnn_scheduler.model import ResidualSchedulingGNN
from gnn_scheduler.configs import Config
from gnn_scheduler.configs.experiment_configs import DEFAULT_CONFIG
from gnn_scheduler.solve_jssp import solve_job_shop_with_gnn
from gnn_scheduler.utils import get_project_path


def load_model(
    model_path: str, config: Optional[Config] = None
) -> ResidualSchedulingGNN:
    """
    Load a trained GNN model from a .pth file.

    Args:
        model_path: Path to the .pth file containing the model weights
        config: Optional config object. If None, DEFAULT_CONFIG will be used

    Returns:
        The loaded model with weights from the .pth file
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG

    # Create model with the same architecture as during training
    model = ResidualSchedulingGNN(
        **config.model_config.to_dict(),
    )

    # Load the model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model


def evaluate_model_performance(
    solver: Callable[[JobShopInstance], Schedule],
    instances: List[JobShopInstance],
    show_progress: bool = True,
) -> tuple[float, float]:
    """
    Evaluate a GNN model's performance on a list of job shop instances by
    calculating the average makespan.

    Args:
        model: The trained GNN model
        instances: List of job_shop_lib.JobShopInstance objects
        show_progress: Whether to show a progress bar during evaluation

    Returns:
        Average makespan of the schedules produced by the model across all
        instances, and the average optimality gap if the optimum is known
    """
    # Create progress bar if requested
    if show_progress:
        instances_iter: Iterable[JobShopInstance] = tqdm.tqdm(
            instances, desc="Evaluating instances"
        )
    else:
        instances_iter = instances

    total_makespan = 0
    total_optimality_gap = 0
    succesful_optimal = 0
    for instance in instances_iter:
        schedule = solver(instance)
        total_makespan += schedule.makespan()
        best_makespan = instance.metadata.get("optimum")
        if "optimum" in instance.metadata:
            best_makespan = instance.metadata["optimum"]
        elif (
            "upper_bound" in instance.metadata
            and "lower_bound" in instance.metadata
        ):
            best_makespan = (
                instance.metadata["upper_bound"]
                + instance.metadata["lower_bound"]
            ) / 2
        if best_makespan is None:
            continue
        optimality_gap = schedule.makespan() / best_makespan
        total_optimality_gap += optimality_gap
        succesful_optimal += 1

    avg_makespan = total_makespan / len(instances)
    avg_optimality_gap = total_optimality_gap / succesful_optimal
    return avg_makespan, avg_optimality_gap


if __name__ == "__main__":
    model = load_model(
        str(get_project_path() / "checkpoints" / "best_model.pth")
    )

    benchmark_instances = load_all_benchmark_instances()

    taillard_instances = [
        instance
        for instance in benchmark_instances.values()
        if "ta" in instance.name
    ]
    gnn_solver = partial(solve_job_shop_with_gnn, model=model)
    # Evaluate the model
    avg_makespan, avg_optimality_gap = evaluate_model_performance(
        gnn_solver, taillard_instances
    )
    print("GNN model performance:")
    print(f"verage makespan: {avg_makespan}")
    print(f"Average optimality gap: {avg_optimality_gap}")

    mwkr_solver = DispatchingRuleSolver()
    avg_makespan, avg_optimality_gap = evaluate_model_performance(
        mwkr_solver, taillard_instances
    )
    print("MWKR performance:")
    print(f"Average makespan: {avg_makespan}")
    print(f"Average optimality gap: {avg_optimality_gap}")

    spt_solver = DispatchingRuleSolver(
        dispatching_rule="shortest_processing_time"
    )
    avg_makespan, avg_optimality_gap = evaluate_model_performance(
        spt_solver, taillard_instances
    )
    print("SPT performance:")
    print(f"Average makespan: {avg_makespan}")
    print(f"Average optimality gap: {avg_optimality_gap}")

    print("FIFO performance:")
    fifo_solver = DispatchingRuleSolver(
        dispatching_rule=DispatchingRuleType.FIRST_COME_FIRST_SERVED
    )

    avg_makespan, avg_optimality_gap = evaluate_model_performance(
        fifo_solver, taillard_instances
    )
    print(f"Average makespan: {avg_makespan}")
    print(f"Average optimality gap: {avg_optimality_gap}")
