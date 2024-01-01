import os
import shutil

from gnn_scheduler import get_data_path


def move_pickle_files(source_folder, destination_folder, file_limit=20000):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Counter for the number of files moved
    count = 0

    # Iterate over files in the source folder
    for file in os.listdir(source_folder):
        if file.endswith(".pkl"):
            # Construct full file path
            source_file = os.path.join(source_folder, file)
            destination_file = os.path.join(destination_folder, file)

            # Move the file
            shutil.move(source_file, destination_file)
            count += 1

            # Stop if the limit is reached
            if count >= file_limit:
                break

    return f"{count} files moved to {destination_folder}"


if __name__ == "__main__":
    # Get the data path
    data_path = get_data_path()

    source_folder_ = (
        data_path
        / "difficulty_prediction"
        / "adj_data_list_diff_prediction_instances"
    )
    destination_folder_ = (
        data_path
        / "difficulty_prediction"
        / "adj_data_list_diff_prediction_instances_2"
    )

    print(move_pickle_files(source_folder_, destination_folder_))
