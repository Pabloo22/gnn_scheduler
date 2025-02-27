import torch
from torch_geometric.data import HeteroData  # type: ignore[import-untyped]


class JobShopGraphData(HeteroData):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "valid_pairs":
            increments = []
            for node_type in ["operation", "machine", "job"]:
                if node_type in self and "x" in self[node_type]:
                    increments.append([self[node_type]["x"].size(0)])
                else:
                    increments.append([0])

            return torch.tensor(increments)

        return super().__inc__(key, value, *args, **kwargs)
