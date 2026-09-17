import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        std = torch.sqrt(torch.tensor(2.0 / (in_features + out_features))).item()
        if weights is None:
            weights = nn.init.trunc_normal_(
                torch.empty([out_features, in_features]),
                std=std,
                a=-3,
                b=3,
            )
        if device is not None:
            weights = weights.to(device)
        self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embedding_matrix: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if embedding_matrix is None:
            # num_embeddings is vocab_size
            embedding_matrix = nn.init.trunc_normal_(
                torch.empty([num_embeddings, embedding_dim]),
                std=1,
                a=-3,
                b=3,
            )
        if device is not None:
            embedding_matrix = embedding_matrix.to(device)
        self.embedding_matrix = nn.Parameter(embedding_matrix, requires_grad=True)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
