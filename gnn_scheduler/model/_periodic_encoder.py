import torch
from torch import nn
import math


class PeriodicEncoder(nn.Module):
    """Periodic encoding layer for scalar inputs.

    Args:
        output_size (int): Desired size of the output embedding
        sigma (float): Standard deviation for initializing frequencies
    """

    def __init__(self, output_size, sigma=1.0):
        super().__init__()
        # Number of frequency components (half of output size since we use both
        # sin and cos)
        self.num_freqs = output_size // 2

        # Initialize trainable frequency multipliers
        # Shape: [num_freqs]
        self.freq_multipliers = nn.Parameter(
            torch.randn((1, self.num_freqs)) * sigma
        )
        self.linear = nn.Linear(self.num_freqs * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms scalar input using periodic encoding.

        Args:
            x (torch.Tensor): Input tensor of shape [..., 1]

        Returns:
            torch.Tensor: Encoded tensor of shape [..., output_size]
        """
        # Ensure x has correct shape
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Compute phase shifts: 2πcx
        # Result shape: [..., num_freqs]
        phase = 2 * math.pi * x * self.freq_multipliers

        # Compute sin and cos components
        sin_components = torch.sin(phase)
        cos_components = torch.cos(phase)

        # Concatenate along last dimension
        # Result shape: [..., 2 * num_freqs]
        encoded = torch.cat([sin_components, cos_components], dim=-1)

        # Apply linear layer
        encoded = self.linear(encoded)
        return encoded


class MultiPeriodicEncoder(nn.Module):
    """Periodic encoding layer for multi-dimensional inputs.

    Args:
        input_size (int): Number of input dimensions
        output_size (int): Desired size of the output embedding
        sigma (float): Standard deviation for initializing frequencies
        concat (bool): Whether to concatenate or sum the encoded dimensions
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        sigma: float = 1.0,
        concat: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.combine_method = concat

        if output_size % input_size != 0:
            raise ValueError("Output size must be divisible by input size")

        feature_output_size = output_size // input_size

        # Number of frequency components per input dimension
        self.num_freqs = feature_output_size // 2

        # Initialize separate trainable frequency multipliers for each input
        # dimension. Shape: [input_size, num_freqs]
        self.freq_multipliers = nn.Parameter(
            torch.randn((input_size, self.num_freqs)) * sigma
        )

        # Create separate linear layers for each input dimension
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(self.num_freqs * 2, feature_output_size)
                for _ in range(input_size)
            ]
        )
        self.concat = concat
        if concat:
            self.final_output_size = feature_output_size * input_size
        else:  # 'sum'
            self.final_output_size = feature_output_size

    @property
    def feature_output_size(self):
        """Size of the output for each input dimension."""
        return self.output_size // self.input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms multi-dimensional input using periodic encoding.

        Args:
            x (torch.Tensor):
                Input tensor of shape [batch_size, input_size] or
                [total_nodes, input_size]

        Returns:
            torch.Tensor:
                Encoded tensor of shape [batch_size, final_output_size] or
                [total_nodes, final_output_size]
        """
        batch_size = x.size(0)

        # Process each input dimension separately
        encoded_dims = []
        for i in range(self.input_size):
            # Get features for current dimension: [batch_size, 1]
            curr_features = x[:, i : i + 1]

            # Compute phase shifts: 2πcx
            # Expand frequency multipliers
            # [1, num_freqs] -> [batch_size, num_freqs]
            curr_freq = self.freq_multipliers[i : i + 1, :].expand(
                batch_size, -1
            )
            phase = 2 * math.pi * curr_features * curr_freq

            # Compute sin and cos components
            sin_components = torch.sin(phase)
            cos_components = torch.cos(phase)

            # Concatenate along frequency dimension
            # Shape: [batch_size, 2 * num_freqs]
            periodic_features = torch.cat(
                [sin_components, cos_components], dim=-1
            )

            # Apply linear transformation
            transformed = self.linear_layers[i](periodic_features)
            encoded_dims.append(transformed)

        if self.concat:
            # Concatenate all encoded dimensions
            # Final shape: [batch_size, input_size * output_size]
            final_encoding = torch.cat(encoded_dims, dim=-1)
        else:  # 'sum'
            # Sum all encoded dimensions
            # Final shape: [batch_size, output_size]
            final_encoding = torch.stack(encoded_dims, dim=0).sum(dim=0)

        return final_encoding


# Example usage:
if __name__ == "__main__":
    # Create encoder with 6-dimensional output
    encoder = MultiPeriodicEncoder(input_size=2, output_size=6, sigma=0.5)

    # Example input (batch_size=2)
    batch = torch.tensor(
        [
            [0.005, 0.005],
            [1, 0.05],
            [1, 0.8],
            [0.9, 1],
        ],
        dtype=torch.float32,
    )

    # Get embeddings
    embeddings = encoder(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embeddings:\n{embeddings}")
