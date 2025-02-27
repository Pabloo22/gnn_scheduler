import torch
from torch_geometric.data import HeteroData


class HeteroEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, hetero_dataset):
        self.dataset = hetero_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Custom collate function to properly batch heterogeneous graphs
def collate_hetero(batch):
    # Initialize empty dictionaries for node features and edge indices
    x_dict_batch = {}
    edge_index_dict_batch = {}
    valid_pairs_batch = []
    y_batch = []

    cumulative_nodes = {node_type: 0 for node_type in batch[0].x_dict.keys()}

    for data in batch:
        # Get node counts for mapping valid_pairs later
        node_counts = {
            node_type: feat.size(0) for node_type, feat in data.x_dict.items()
        }

        # Add node features to batch
        for node_type, features in data.x_dict.items():
            if node_type not in x_dict_batch:
                x_dict_batch[node_type] = [features]
            else:
                x_dict_batch[node_type].append(features)

        # Add edge indices to batch with proper node offset
        for edge_key, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_key

            # Adjust indices based on cumulative node counts
            edge_index_offset = edge_index.clone()
            edge_index_offset[0] += cumulative_nodes[src_type]
            edge_index_offset[1] += cumulative_nodes[dst_type]

            if edge_key not in edge_index_dict_batch:
                edge_index_dict_batch[edge_key] = [edge_index_offset]
            else:
                edge_index_dict_batch[edge_key].append(edge_index_offset)

        # Handle valid_pairs
        valid_pairs_offset = data.valid_pairs.clone()
        valid_pairs_offset[:, 0] += cumulative_nodes["operation"]
        valid_pairs_offset[:, 1] += cumulative_nodes["machine"]
        valid_pairs_offset[:, 2] += cumulative_nodes["job"]

        valid_pairs_batch.append(valid_pairs_offset)

        if hasattr(data, "y"):
            y_batch.append(data.y)

        # Update cumulative node counts
        for node_type, count in node_counts.items():
            cumulative_nodes[node_type] += count

    # Concatenate all features and edges
    for node_type in x_dict_batch:
        x_dict_batch[node_type] = torch.cat(x_dict_batch[node_type], dim=0)

    for edge_key in edge_index_dict_batch:
        edge_index_dict_batch[edge_key] = torch.cat(
            edge_index_dict_batch[edge_key], dim=1
        )

    # Concatenate valid_pairs if they exist
    valid_pairs_batch = torch.cat(valid_pairs_batch, dim=0)

    if y_batch:
        y_batch = torch.cat(y_batch, dim=0)

    # Create a new HeteroData object with the batched data
    batched_data = HeteroData()
    batched_data.x_dict = x_dict_batch
    batched_data.edge_index_dict = edge_index_dict_batch
    batched_data.valid_pairs = valid_pairs_batch

    return batched_data


# # Create dataset and dataloader
# dataset = HeteroEdgeDataset(hetero_dataset)
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=32,  # Adjust as needed
#     shuffle=True,  # Set to False for validation/testing
#     collate_fn=collate_hetero,
# )

# # Usage:
# for batch in dataloader:
#     # batch is a HeteroData object with batched data
#     # You can access batch.x_dict, batch.edge_index_dict, batch.valid_pairs
#     # Forward pass to your model
#     out = model(batch.x_dict, batch.edge_index_dict, batch.valid_pairs)
#     # Rest of your training loop
