import pandas as pd
import random
from job_shop_lib.dispatching.rules import (
    DispatchingRuleSolver,
    DispatchingRuleType,
)
from job_shop_lib.benchmarking import load_all_benchmark_instances
from gnn_scheduler.eval import get_performance_dataframe
from gnn_scheduler.utils import get_data_path


def _main():
    # Load all benchmark instances
    instances = load_all_benchmark_instances().values()

    # Create a dispatching rule solver
    spt_solver = DispatchingRuleSolver(
        dispatching_rule=DispatchingRuleType.SHORTEST_PROCESSING_TIME
    )

    # Evaluate the dispatching rule solver on the benchmark instances
    performance_df = get_performance_dataframe(
        solver=spt_solver,
        instances=instances,
    )
    # Rename "makespan" and "optimization_gap" columns
    performance_df = performance_df.rename(
        columns={
            "makespan": "makespan_spt",
            "optimality_gap": "optimality_gap_spt",
        }
    )
    mwkr_solver = DispatchingRuleSolver()
    mwkr_df = get_performance_dataframe(
        solver=mwkr_solver,
        instances=instances,
    )
    # Rename "makespan" and "optimization_gap" columns
    mwkr_df = mwkr_df[["makespan", "optimality_gap"]].rename(
        columns={
            "makespan": "makespan_mwkr",
            "optimality_gap": "optimality_gap_mwkr",
        }
    )

    mor_solver = DispatchingRuleSolver(
        dispatching_rule=DispatchingRuleType.MOST_OPERATIONS_REMAINING
    )
    mor_df = get_performance_dataframe(
        solver=mor_solver,
        instances=instances,
    )
    # Rename "makespan" and "optimization_gap" columnss
    mor_df = mor_df[["makespan", "optimality_gap"]].rename(
        columns={
            "makespan": "makespan_mor",
            "optimality_gap": "optimality_gap_mor",
        }
    )
    random.seed(42)
    random_solver = DispatchingRuleSolver(
        dispatching_rule=DispatchingRuleType.RANDOM
    )
    random_df = get_performance_dataframe(
        solver=random_solver,
        instances=instances,
    )

    # Rename "makespan" and "optimization_gap" columns
    random_df = random_df[["makespan", "optimality_gap"]].rename(
        columns={
            "makespan": "makespan_random",
            "optimality_gap": "optimality_gap_random",
        }
    )

    # FIFO
    fifo_solver = DispatchingRuleSolver(
        dispatching_rule=DispatchingRuleType.FIRST_COME_FIRST_SERVED
    )
    fifo_df = get_performance_dataframe(
        solver=fifo_solver,
        instances=instances,
    )
    # Rename "makespan" and "optimization_gap" columns
    fifo_df = fifo_df[["makespan", "optimality_gap"]].rename(
        columns={
            "makespan": "makespan_fifo",
            "optimality_gap": "optimality_gap_fifo",
        }
    )

    # Merge all dataframes
    all_dfs = [
        performance_df,
        mwkr_df,
        mor_df,
        random_df,
        fifo_df,
    ]
    merged_df = pd.concat(all_dfs, axis=1)
    print(merged_df)

    # Save the merged dataframe to a .csv file
    merged_df.to_csv(
        get_data_path() / "dispatching_rule_performance.csv",
        index=False,
    )


if __name__ == "__main__":
    _main()
