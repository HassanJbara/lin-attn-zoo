import torch
import torch.nn as nn


class GatedRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-8,
        activation="swish",
        use_weight=True,
        use_bias=False,
    ):
        """
        Args:
            normalized_shape (int or tuple): input shape from an expected input.
                If a single integer, it is treated as a singleton tuple.
            eps (float): a value added to the denominator for numerical stability.
            activation (str): the activation to use for gating. Options: 'swish' (or 'silu') and 'sigmoid'.
            use_weight (bool): If True, apply a learnable scale after normalization.
            use_bias (bool): If True, apply a learnable bias after normalization.
        """
        super(GatedRMSNorm, self).__init__()
        assert activation.lower() in ["swish", "silu", "sigmoid"], (
            f"Unsupported activation type: {activation}"
        )
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.activation = activation.lower()

        if use_weight:
            # Learnable scale parameter applied elementwise
            self.weight = nn.Parameter(torch.ones(*normalized_shape))
        else:
            self.register_parameter("weight", None)

        if use_bias:
            # Learnable bias parameter applied elementwise
            self.bias = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, g, residual=None) -> torch.Tensor:
        x = x + residual if residual is not None else x

        # Compute RMS normalization (skip mean subtraction)
        # Calculate mean of squared values over the last dimension(s) that are normalized
        # Here, we assume normalization is over the last len(normalized_shape) dimensions.
        dims = tuple(range(-len(self.normalized_shape), 0))
        rms = torch.sqrt(torch.mean(x.pow(2), dim=dims, keepdim=True) + self.eps)
        x_hat = x / rms

        if self.weight is not None:
            x_hat = x_hat * self.weight
        if self.bias is not None:
            x_hat = x_hat + self.bias

        if self.activation in ["swish", "silu"]:
            gated = g * torch.sigmoid(g)
        else:
            gated = torch.sigmoid(g)

        return x_hat * gated


class Swish(nn.Module):
    def forward(self, x) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        swish = self.linear1(x) * self.linear1(x).sigmoid()
        gate = self.linear2(x)
        return swish * gate
