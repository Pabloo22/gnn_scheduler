from collections.abc import Iterable, Callable
from collections import defaultdict
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
import pandas as pd

from gnn_scheduler.model import ResidualSchedulingGNN
from gnn_scheduler.configs import Config
from gnn_scheduler.configs.experiment_configs import EXPERIMENT_2
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
    # Handle different formats of saved weights
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model


def evaluate_model_performance(
    solver: Callable[[JobShopInstance], Schedule],
    instances: List[JobShopInstance],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Evaluate a model's performance on a list of job shop instances.

    Args:
        solver: Function that takes a JobShopInstance and returns a Schedule
        instances: List of job_shop_lib.JobShopInstance objects
        show_progress: Whether to show a progress bar during evaluation

    Returns:
        A pandas DataFrame with detailed results for each instance
    """
    # Create progress bar if requested
    if show_progress:
        instances_iter: Iterable[JobShopInstance] = tqdm.tqdm(
            instances, desc="Evaluating instances"
        )
    else:
        instances_iter = instances

    # Lists to collect data for DataFrame
    instance_names = []
    job_counts = []
    machine_counts = []
    makespans = []
    optimality_gaps = []
    optimums = []
    upper_bounds = []
    lower_bounds = []

    # Also track total stats for backward compatibility
    total_makespan = 0
    total_optimality_gap = 0
    successful_optimal = 0

    for instance in instances_iter:
        schedule = solver(instance)
        makespan = schedule.makespan()

        # Add to overall statistics
        total_makespan += makespan
        # Collect data for DataFrame
        instance_names.append(instance.name)
        job_counts.append(instance.num_jobs)
        machine_counts.append(instance.num_machines)
        makespans.append(makespan)

        # Extract reference values from metadata
        optimum = instance.metadata.get("optimum")
        upper_bound = instance.metadata.get("upper_bound")
        lower_bound = instance.metadata.get("lower_bound")

        optimums.append(optimum)
        upper_bounds.append(upper_bound)
        lower_bounds.append(lower_bound)

        # Calculate optimality gap if possible
        best_makespan = None
        if optimum is not None:
            best_makespan = optimum
        elif upper_bound is not None and lower_bound is not None:
            best_makespan = (upper_bound + lower_bound) / 2

        if best_makespan is not None:
            gap = makespan / best_makespan
            optimality_gaps.append(gap)
            total_optimality_gap += gap
            successful_optimal += 1
        else:
            optimality_gaps.append(None)
    results_df = pd.DataFrame(
        {
            "instance_name": instance_names,
            "num_jobs": job_counts,
            "num_machines": machine_counts,
            "makespan": makespans,
            "optimality_gap": optimality_gaps,
            "optimum": optimums,
            "upper_bound": upper_bounds,
            "lower_bound": lower_bounds,
        }
    )
    return results_df


if __name__ == "__main__":
    model = load_model(
        str(get_project_path() / "checkpoints" / "best_model.pth"),
        config=EXPERIMENT_2,
    )

    benchmark_instances = load_all_benchmark_instances()

    taillard_instances = [
        instance
        for instance in benchmark_instances.values()
        if "ta" in instance.name
    ][:10]
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
