from gnn_scheduler.difficulty_prediction import (
    process_data,
    save_adj_data_list,
    process_and_save_data,
)

FOLDER_NAMES = ["augmented_benchmark_10machines", "diff_prediction_instances"]

NEW_FOLDER_NAMES = [
    "adj_data_list_augmented_benchmark_10machines",
    "adj_data_list_diff_prediction_instances",
]


def main():
    adj_data_list_augmented_benchmark_10machines = process_data(
        [FOLDER_NAMES[0]],
        show_progress=True,
        sparse=False,
        directed=False,
    )
    save_adj_data_list(
        adj_data_list_augmented_benchmark_10machines,
        folder_name=NEW_FOLDER_NAMES[0],
        show_progress=True,
    )

    # adj_list_diff_prediction_instances = process_data(
    #     [FOLDER_NAMES[1]],
    #     show_progress=True,
    #     sparse=False,
    #     directed=False,
    # )

    # save_adj_data_list(
    #     adj_list_diff_prediction_instances,
    #     folder_name=NEW_FOLDER_NAMES[1],
    #     show_progress=True,
    # )

    # For this folder we save the data in batches to avoid memory issues
    process_and_save_data(
        [FOLDER_NAMES[1]],
        [NEW_FOLDER_NAMES[1]],
        show_progress=True,
        sparse=False,
        directed=False,
        batch_size=1000,
    )


if __name__ == "__main__":
    main()
