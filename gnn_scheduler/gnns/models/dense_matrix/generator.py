import torch
from torch import nn
import torch.nn.functional as F

from gnn_scheduler.gnns.models.dense_matrix import MultiDenseLayer
from gnn_scheduler.gnns import get_activation_function


class Generator(nn.Module):
    def __init__(
        self,
        max_nodos: int,
        num_machines: int,
        hidden_dims: list[int],
        activation: str | None = "relu",
        dropout_rate: float = 0.2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_nodos = max_nodos
        self.num_machines = num_machines
        self.temperature = temperature
        activation = get_activation_function(activation)
        self.mlp = MultiDenseLayer(
            max_nodos,
            hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.final_layer = nn.Linear(hidden_dims[-1], max_nodos * num_machines)

    def forward(self, z: torch.Tensor, n_valid_nodes: int):
        h = self.mlp(z)
        h = h[:n_valid_nodes, :]
        logits = self.final_layer(h).view(-1, self.num_machines)

        # Apply Gumbel-Softmax
        x = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=1)

        disjunctive_adj_matrix = self.compute_adjacency(x)
        return x, disjunctive_adj_matrix

    def compute_adjacency(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        disjunctive_adj_matrix = torch.matmul(x, x.t())

        # Remove self-loops
        disjunctive_adj_matrix = disjunctive_adj_matrix - torch.diag(
            torch.diag(disjunctive_adj_matrix)
        )
        return disjunctive_adj_matrix
