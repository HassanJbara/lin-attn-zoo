import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        swish = self.linear1(x) * self.linear1(x).sigmoid()
        gate = self.linear2(x)
        return swish * gate
