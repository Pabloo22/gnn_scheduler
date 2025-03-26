import os
import json
import numpy as np
from gnn_scheduler.utils import get_data_path


def main():
    data_path_raw = get_data_path() / "raw"

    files = os.listdir(data_path_raw)
    files_with_train_in_the_name = [file for file in files if "train" in file]
    instances_matrices = []
    for file in files_with_train_in_the_name:
        with open(data_path_raw / file, "r", encoding="utf-8") as f:
            data = json.load(f)
            schedule_dict = data[0]
            instance_dict = schedule_dict["instance"]
            duration_matrix = np.array(instance_dict["duration_matrix"])
            for i, instance_matrix in enumerate(instances_matrices):
                if np.array_equal(duration_matrix, instance_matrix):
                    print(
                        f"{file} is a duplicate of "
                        f"{files_with_train_in_the_name[i]}"
                    )
            instances_matrices.append(duration_matrix)

    print("Processing complete.")


if __name__ == "__main__":
    main()
