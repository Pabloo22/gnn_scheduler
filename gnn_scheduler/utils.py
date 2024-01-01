import pathlib
import os

import dotenv  # pylint: disable=import-error


def get_project_path() -> pathlib.Path:
    """Gets project path."""
    dotenv.load_dotenv()
    project_path = os.getenv("PROJECT_PATH")

    if project_path is None:
        raise ValueError("PROJECT_PATH environment variable not set.")

    return pathlib.Path(project_path)


def get_data_path() -> pathlib.Path:
    """Returns project/path/data."""
    return get_project_path() / "data"


if __name__ == "__main__":
    print(get_project_path())
