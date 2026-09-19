```python

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        weights: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        if weights is None:
            self.g = nn.Parameter(torch.ones(d_model), requires_grad=True)
        else:
            self.g = nn.Parameter(weights, requires_grad=True)

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((torch.square(x).sum(dim=-1, keepdim=True) / self.d_model) + self.eps)
        # rmsnorm = einsum(x, self.g,"... d_in, d_in -> ... d_in")/rms
        rmsnorm = torch.mul(x, self.g) / rms
        return rmsnorm.to(in_dtype)
```
Here what was doing initially => torch.square(x).sum()
This made the (b, seq, d_model) vector a vector of shape 1

> Solution : torch.square(x).sum(dim=-1)
    This turned the vector to shape (b, seq)

**But we still want the last dimension as otherwise in rmsnorm calculation when dividing by rms there would be a shape mismatch**

> Solution : torch.square(x).sum(dim=-1, keepdim=True)
    This would keep the shape as (b, seq, d_model)

## Handling devices and dtype
```python
        if weight is None:
            weight = nn.Parameter(torch.ones(d_model), requires_grad=True)
        else:
            weight = nn.Parameter(weight, requires_grad=True)
        if device is not None:
            weight = weight.to(device=device)
        if dtype is not None:
            weight = weight.to(dtype=dtype)
        self.weight = weight
```
Have a look at the code above. First a parameter is created out of weights, and then it is set to the correct device and dtype. 
> ***Using .to on a parameter changes it to a tensor***
