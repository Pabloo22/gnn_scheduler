import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gnn_scheduler.utils import get_data_path


def main(csv_name: str, only_taillard: bool = False, model_name: str = "HGIN"):
    data_path = get_data_path()
    dpr_df = pd.read_csv(data_path / "dispatching_rule_performance.csv")
    # keep columns that start with "makespan" or "optimality_gap"
    dpr_df = dpr_df[
        [
            col
            for col in dpr_df.columns
            if col.startswith("makespan") or col.startswith("optimality_gap")
        ]
    ]
    model_df = pd.read_csv(data_path / f"{csv_name}.csv")

    # Merge the two dataframes
    merged_df = pd.concat([model_df, dpr_df], axis=1)

    # Substract one to all optimality gap columns to get the optimality gap
    optimality_gap_cols = [
        col for col in merged_df.columns if col.startswith("optimality_gap")
    ]
    for col in optimality_gap_cols:
        merged_df[col] -= 1

    # remove ft06
    merged_df = merged_df[merged_df["instance_name"] != "ft06"]

    if only_taillard:
        # keep only taillard instances
        merged_df = merged_df[merged_df["instance_name"].str.startswith("ta")]

    # Drop name column
    merged_df_agg = merged_df.drop(columns=["instance_name"])
    # aggregate by problem size
    merged_df_agg = (
        merged_df_agg.groupby(["num_jobs", "num_machines"])
        .mean()
        .reset_index()
    )
    print(merged_df_agg)
    only_taillard_str = "_only_taillard" if only_taillard else ""
    merged_df_agg.to_csv(
        data_path / f"{csv_name}{only_taillard_str}_with_dpr.csv",
        index=False,
    )

    # Create bar plot comparing the optimality gap of the model and each of
    # the dispatching rules for each problem size
    create_optimality_gap_plot(
        merged_df_agg, csv_name, data_path, only_taillard_str, model_name
    )


def create_optimality_gap_plot(
    df, csv_name, data_path, only_taillard_str, model_name="HGIN"
):
    # Get all the optimality gap columns
    optimality_cols = [
        col for col in df.columns if col.startswith("optimality_gap")
    ]

    # Set up the problem sizes for x-axis
    problem_sizes = []
    for _, row in df.iterrows():
        problem_sizes.append(
            f"{int(row['num_jobs'])}x{int(row['num_machines'])}"
        )

    # Set up the figure and axis
    plt.figure(figsize=(14, 8))

    # Number of bars and their positions
    n_bars = len(optimality_cols)
    bar_width = 0.8 / n_bars
    index = np.arange(len(problem_sizes))

    # Define a color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Create bars for each optimality gap
    for i, col in enumerate(optimality_cols):
        # Use a readable label: remove 'optimality_gap_' prefix and capitalize
        if col == "optimality_gap":
            label = f"{model_name} (Ours)"
        else:
            label = col.replace("optimality_gap_", "").upper()

        # Plot the bars
        plt.bar(
            index + i * bar_width - (n_bars * bar_width / 2) + bar_width / 2,
            df[col],
            bar_width,
            label=label,
            color=colors[i % len(colors)],
        )

    # Customize the plot
    plt.xlabel("Problem Size (Jobs × Machines)", fontsize=12)
    plt.ylabel("Optimality Gap (%)", fontsize=12)
    plt.title(
        "Comparison of Optimality Gap by Problem Size and Method", fontsize=14
    )
    plt.xticks(index, problem_sizes, rotation=45)
    plt.legend(loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        data_path
        / f"{csv_name}{only_taillard_str}_optimality_gap_comparison.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    main("experiment22_results", only_taillard=True, model_name="HGIN")
