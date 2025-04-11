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
| Mixture of Memories | | | |
