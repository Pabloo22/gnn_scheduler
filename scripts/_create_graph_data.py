import torch.serialization
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
from gnn_scheduler.data import JobShopDataset, JobShopData


# Add JobShopData to PyTorch's safe globals to fix the deserialization warning
torch.serialization.add_safe_globals(
    [JobShopData, BaseStorage, NodeStorage, EdgeStorage]
)


if __name__ == "__main__":
    dataset = JobShopDataset(num_chunks=20, force_reload=True)
