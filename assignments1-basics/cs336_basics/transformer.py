from math import exp

from numpy import dtype
import torch
from torch import nn
from einops import einsum, rearrange


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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.theta = torch.tensor(theta)
        if device is not None:
            self.theta = self.theta.to(device)
        # weight = torch.tensor(torch.zeros(max_seq_len, d_k//2))
        # for loops not good
        # for i in range(weight.size(dim=-2)):
        #     for j in range(1, weight.size(dim=-1)+1):
        #         weight[i][j-1] = i/torch.pow(self.theta, (2*j-2)/d_k)

        # vectorized implementation
        positions = torch.arange(0, max_seq_len, device=device)
        frequency = torch.pow(self.theta, -torch.arange(0, d_k, 2, device=device) / d_k)
        weight = einsum(positions, frequency, "max_seq_len, v -> max_seq_len v")
        self.register_buffer("cos", torch.cos(weight), persistent=False)
        self.register_buffer("sin", torch.sin(weight), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        y = rearrange(x, "... seq_len (d two) -> ... seq_len d two", two=2)
        sin = self.sin[token_positions]  # shape => seq_len, d_k/2
        cos = self.cos[token_positions]
        y0 = y[..., 0] * cos - y[..., 1] * sin
        y1 = y[..., 0] * sin + y[..., 1] * cos
        y = torch.stack([y0, y1], dim=-1)
        y = rearrange(y, "... seq_len d two -> ... seq_len (d two)")

        return y


def softmax(x: torch.Tensor, dim: int):
    x_max = torch.max(x, dim=dim, keepdim=True)
    x_c = x - x_max.values
    exp_x_c = torch.exp(x_c)
    denom = torch.sum(exp_x_c, dim=dim, keepdim=True)
    return exp_x_c / denom


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    affinities = einsum(Q, K, "... n d,... m d ->... n m") / Q.size(dim=-1) ** 0.5
    if mask is not None:
        mask_affinities = torch.where(mask, affinities, -torch.inf)
    else:
        mask_affinities = affinities
    softmax_masked_affinities = softmax(mask_affinities, dim=-1)
    out = einsum(softmax_masked_affinities, V, "... n m,... m d_k ->... n d_k")
    return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        Q: torch.Tensor | None = None,
        K: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
        O: torch.Tensor | None = None,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        if max_seq_len is not None:
            self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        else:
            self.rope = None
        # Q, K, V are d_model, d_model
        self.q_proj = Linear(d_model, d_model, Q)
        self.k_proj = Linear(d_model, d_model, K)
        self.v_proj = Linear(d_model, d_model, V)
        self.output_proj = Linear(d_model, d_model, O)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "... sl (h d_k) -> ... h sl d_k", h=self.num_heads)
        k = rearrange(k, "... sl (h d_k) -> ... h sl d_k", h=self.num_heads)
        v = rearrange(v, "... sl (h d_k) -> ... h sl d_k", h=self.num_heads)
        if self.max_seq_len is None:
            self.max_seq_len = x.size(dim=-2)
            self.mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, device=x.device, dtype=torch.bool))
        self.mask = self.mask.to(x.device)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0, x.size(dim=-2), device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        affinities = scaled_dot_product_attention(q, k, v, self.mask[: x.size(dim=-2), : x.size(dim=-2)])
        affinities = rearrange(affinities, "... h sl d_k -> ... sl (h d_k)")
        o = self.output_proj(affinities)
        return o


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for layer in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor):
        y = self.token_embeddings(in_indices)
        for l in self.layers:
            y = l(y)
        y = self.ln_final(y)
        y = self.lm_head(y)
        return y
