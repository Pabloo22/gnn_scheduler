import os

import tqdm
from torch_geometric.data import InMemoryDataset
from gnn_scheduler.data import JobShopData, DatasetManager
from gnn_scheduler.utils import get_data_path


class CombinedJobShopDataset(InMemoryDataset):
    """Dataset class that combines multiple JobShopDataset instances into a
    single dataset.

    This class leverages the DatasetManager to access individual datasets and
    combines them into a single dataset for more efficient training.
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        root=None,
        transform=None,
        pre_transform=None,
        force_reload=False,
        log=True,
        processed_filename="combined_dataset.pt",
    ):
        self.dataset_manager = dataset_manager
        self.processed_filename = processed_filename

        # Use a dedicated directory for the combined dataset
        if root is None:
            root = str(get_data_path())

        super().__init__(
            root, transform, pre_transform, force_reload=force_reload, log=log
        )
        self.load(self.processed_paths[0], data_cls=JobShopData)

    @property
    def raw_file_names(self):
        # We don't have raw files, just return an empty list
        return []

    @property
    def processed_file_names(self):
        # Just one combined file will be created
        return [self.processed_filename]

    def download(self):
        # No download needed
        pass

    def process(self):
        """Process the dataset by combining multiple datasets from
        DatasetManager.

        This method iterates over the dataset_manager to access each individual
        dataset, extracts all data objects, and combines them into a single
        dataset.
        """
        all_data = []
        total_count = 0

        # Iterate over the DatasetManager to get each dataloader
        for i, dataloader in enumerate(self.dataset_manager, start=1):
            if self.log:
                print(f"Processing dataset {i}/{len(self.dataset_manager)}")
            dataset = dataloader.dataset

            if self.log:
                print(f"Processing dataset with {len(dataset)} examples")

            # Get all data objects from this dataset
            for j in tqdm.tqdm(
                range(len(dataset)),
                desc="Processing data",
                disable=not self.log,
            ):
                assert isinstance(dataset, InMemoryDataset)
                data = dataset.get(j)
                all_data.append(data)

            total_count += len(dataset)

        if self.log:
            print(f"Combined dataset created with {total_count} examples")

        # Save the combined dataset
        os.makedirs(self.processed_dir, exist_ok=True)
        self.save(all_data, self.processed_paths[0])
