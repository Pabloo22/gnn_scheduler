import torch
from torch_geometric.data import HeteroData  # type: ignore[import-untyped]
from torch_geometric.data.storage import (  # type: ignore[import-untyped]
    BaseStorage,
    NodeStorage,
    EdgeStorage,
)


class JobShopData(HeteroData):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "valid_pairs":
            assert isinstance(value, torch.Tensor)
            dim_size = value.size(1)
            increments = torch.zeros(
                dim_size, dtype=torch.long, device=value.device
            )

            # Set increments for each column according to respective node
            # counts
            node_types = ["operation", "machine", "job"]
            for i, node_type in enumerate(node_types):
                if self[node_type]:
                    increments[i] = self[node_type]["x"].size(0)

            return increments

        return super().__inc__(key, value, *args, **kwargs)


torch.serialization.add_safe_globals(
    [
        JobShopData,
        HeteroData,
        BaseStorage,
        NodeStorage,
        EdgeStorage,
    ]
)
