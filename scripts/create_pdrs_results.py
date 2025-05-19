import pandas as pd
import random
from job_shop_lib.dispatching.rules import (
    DispatchingRuleSolver,
    DispatchingRuleType,
)
from job_shop_lib.benchmarking import load_all_benchmark_instances
from gnn_scheduler.eval import get_performance_dataframe
from gnn_scheduler.utils import get_data_path


def main():
    """
    Tests all dispatching rules supported by job_shop_lib and saves the results
    in a CSV file named pdrs_results.csv in the DATA_PATH.
    Columns of this dataset are "instance_name" and makespan_{PDR_ALIAS}
    (e.g., makespan_MWKR).
    """
    instances = load_all_benchmark_instances().values()
    all_pdrs = list(DispatchingRuleType)

    # Define mapping of PDR types to their shortened aliases
    pdr_mapping = {
        DispatchingRuleType.MOST_WORK_REMAINING: "MWKR",
        DispatchingRuleType.SHORTEST_PROCESSING_TIME: "SPT",
        DispatchingRuleType.MOST_OPERATIONS_REMAINING: "MOR",
        DispatchingRuleType.FIRST_COME_FIRST_SERVED: "FCFS",
        DispatchingRuleType.RANDOM: "random",
    }

    results_df = None

    # Process all PDRs except RANDOM first
    for pdr_type in all_pdrs:
        if pdr_type == DispatchingRuleType.RANDOM:
            continue

        solver = DispatchingRuleSolver(dispatching_rule=pdr_type)
        performance_df = get_performance_dataframe(
            solver=solver,
            instances=instances,
        )

        # Get the shortened alias for this PDR
        pdr_alias = pdr_mapping.get(pdr_type, pdr_type.name)

        performance_df = performance_df.rename(
            columns={"makespan": f"makespan_{pdr_alias}"}
        )

        # Keep only instance_name and the makespan column
        current_results_df = performance_df[
            ["instance_name", f"makespan_{pdr_alias}"]
        ]

        if results_df is None:
            results_df = current_results_df
        else:
            results_df = pd.merge(
                results_df, current_results_df, on="instance_name", how="outer"
            )

    # Handle RANDOM PDR separately - run 5 times and average the results
    print("Running RANDOM PDR 5 times and averaging results...")
    random_dfs = []
    for i in range(5):
        print(f"RANDOM run {i+1}/5")
        # Set different random seed for each run
        random.seed(42 + i)
        random_solver = DispatchingRuleSolver(
            dispatching_rule=DispatchingRuleType.RANDOM
        )
        random_df = get_performance_dataframe(
            solver=random_solver,
            instances=instances,
        )
        random_dfs.append(random_df[["instance_name", "makespan"]])

    # Combine all random results by instance_name and average the makespan
    random_combined = (
        pd.concat(random_dfs).groupby("instance_name").mean().reset_index()
    )
    random_combined = random_combined.rename(
        columns={"makespan": "makespan_random"}
    )

    # Merge with the main results dataframe
    if results_df is None:
        results_df = random_combined
    else:
        results_df = pd.merge(
            results_df, random_combined, on="instance_name", how="outer"
        )

    if results_df is not None:
        output_path = get_data_path() / "pdrs_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
