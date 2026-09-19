import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        std = torch.sqrt(torch.tensor(2.0 / (in_features + out_features))).item()
        if weight is None:
            weight = nn.init.trunc_normal_(
                torch.empty([out_features, in_features]),
                std=std,
                a=-3,
                b=3,
            )
        if device is not None:
            weight = weight.to(device)
        if dtype is not None:
            weight = weight.to(dtype=dtype)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if weight is None:
            # num_embeddings is vocab_size
            weight = nn.init.trunc_normal_(
                torch.empty([num_embeddings, embedding_dim]),
                std=1,
                a=-3,
                b=3,
            )
        if device is not None:
            weight = weight.to(device)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        weight: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        if weight is None:
            weight = torch.ones(d_model)
        if device is not None:
            weight = weight.to(device=device)
        if dtype is not None:
            weight = weight.to(dtype=dtype)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((torch.square(x).sum(dim=-1, keepdim=True) / self.d_model) + self.eps)
        rmsnorm = torch.mul(x, self.weight) / rms
        # rmsnorm = einsum(x, self.weight,"... d_in, d_in -> ... d_in")/rms
        return rmsnorm.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.mul(x, torch.sigmoid(x))


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        W1: torch.Tensor | None = None,
        W2: torch.Tensor | None = None,
        W3: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, W1, device, dtype)
        self.w2 = Linear(d_ff, d_model, W2, device, dtype)
        self.w3 = Linear(d_model, d_ff, W3, device, dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor):
        out = self.silu(self.w1(x))
        out = torch.mul(out, self.w3(x))
        return self.w2(out)
