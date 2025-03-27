import json
import os

from gnn_scheduler.utils import get_data_path


def divide_data_list(data_list, num_splits):
    split_size = len(data_list) // num_splits
    splits = []
    for i in range(num_splits):
        splits.append(data_list[i * split_size : (i + 1) * split_size])
    return splits


def _main(json_filename: str, num_splits=2):
    json_path = str(get_data_path() / "raw")
    json_filename = os.path.join(json_path, json_filename)
    with open(json_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    splits = divide_data_list(data, num_splits)
    json_filename_stem = json_filename.split(".")[0]
    for i, split in enumerate(splits):
        with open(
            f"{json_filename_stem}_{i}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(split, f)

    # Remove the original file
    os.remove(json_filename)


if __name__ == "__main__":
    from gnn_scheduler.configs.experiment_configs import (
        TRAIN_JSONS_WITHOUT_8X8,
    )

    for train_json in TRAIN_JSONS_WITHOUT_8X8:
        _main(train_json)
