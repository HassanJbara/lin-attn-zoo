# Linear Attention Zoo

Simple, pure PyTorch implementation of popular linear attention models for demonstration and ease of use. Just `pip install torch` to get started!

## Motivation

1. **Pure PyTorch:** No platform-specific libraries (e.g. Triton) or extra dependencies.
2. **Simple**: No complex abstractions or interdependencies, one or two files per model.
3. **No Optimization**: No GPU kernels or memory efficiency, just demonstration of linear attention concepts.

## Models

| Model | Paper | Code | Official Implementation |
|-------|-------|------| ---------------------|
| DeltaNet |  | [deltanet.py](models/deltanet.py) | |
| Gated DeltaNet | |[gated_deltanet.py](models/gated_deltanet.py) | |
| DeltaProduct | | | |

## Speed

While this implementation is not optimized for speed, it doesn't necessarily need to be slow. One advantage of having a pure PyTorch implementation is that it can take better advantage of PyTorch's features, such as compilation. For example, you can use `torch.compile` to speed up the models:

```python
import pytest
import torch
import time
from fla.layers import DeltaNet as DeltaNetFLA
from models.deltanet import DeltaNet

def benchmark_speedup(
        B: int,
        T: int,
        H: int,
        dtype: torch.dtype
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_heads = 2

    model1 = DeltaNet(
        mode="chunk",
        hidden_size=H,
        num_heads=n_heads,
        chunk_size=64,
    ).to(dtype).to(device)

    model2 = DeltaNetFLA(
        mode="chunk",
        hidden_size=H,
        num_heads=n_heads,
        chunk_size=64,
    ).to(dtype).to(device)

    model2.load_state_dict(model1.state_dict())
    model1.eval()
    model2.eval()
    model1 = torch.compile(model1, mode="max-autotune", fullgraph=True)

    def benchmark_model(model, num_runs=100):
      # Warmup
      for _ in range(3):
          x = torch.randn(B, T, H).to(dtype).to(device).requires_grad_(True)
          with torch.no_grad():
              model(x)
      
      # Benchmark
      torch.cuda.synchronize() if torch.cuda.is_available() else None
      start_time = time.time()
      for _ in range(num_runs):
          x = torch.randn(B, T, H).to(dtype).to(device).requires_grad_(True)
          with torch.no_grad():
              model(x)
      torch.cuda.synchronize() if torch.cuda.is_available() else None
      end_time = time.time()
      
      return (end_time - start_time) / num_runs  # Average time per run

    # Run benchmarks
    original_time = benchmark_model(model2)
    compiled_time = benchmark_model(model1)

    print(f"Original model: {original_time:.6f} seconds per run")
    print(f"Compiled model: {compiled_time:.6f} seconds per run")
    print(f"Speedup: {original_time / compiled_time:.2f}x")

if __name__ == "__main__":
    benchmark_speedup(B=2, T=512, H=128, dtype=torch.float16)
```

This yields a speedup of anything between 1.1x and 1.6x, depending on the device and model/input size. This is probably due to full graph compilation and more aggressive fusing that PyTorch can do when there are no graph breaking operations present, such as custom kernels. This obviously lacks the nice features of other implementations, such as caching, but the fact that it matches or surpasses the speed of custom kernels is surprising.

More speed benchmarking to come.
