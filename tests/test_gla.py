import pytest
import torch

# Try to import FLA version
try:
    from fla.layers import GatedLinearAttention as GatedLinearAttentionFLA

    FLA_AVAILABLE = True
except Exception as e:
    print(f"FLA not available for GLA: {e}")
    FLA_AVAILABLE = False


# global variables for testing
BATCH_SIZES = [1, 3]
SEQ_LENGTHS = [16, 32]
HIDDEN_SIZES = [64]
NUM_HEADS = [4]
DTYPES = [torch.float32, torch.float16]


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gla_forward_shapes(B, T, H, num_heads, dtype):
    gla = GatedLinearAttention(mode="recurrent", hidden_size=H, num_heads=num_heads).to(
        dtype
    )
    x = torch.randn(B, T, H, dtype=dtype)
    o, state = gla(x)

    # output should be [B, T, H]
    assert o.shape == (B, T, H)
    head_dim = H // num_heads
    # state should be [B, 1, num_heads, head_dim, head_dim]
    assert state.shape == (B, 1, num_heads, head_dim, head_dim)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
def test_gla_backpropagation(B, T, H, num_heads):
    gla = GatedLinearAttention(mode="recurrent", hidden_size=H, num_heads=num_heads).to(
        torch.float32
    )
    gla.train()
    optimizer = torch.optim.Adam(gla.parameters(), lr=1e-3)
    optimizer.zero_grad()

    x = torch.randn(B, T, H, dtype=torch.float32)
    out, state = gla(x)
    loss = out.sum()
    assert not torch.isnan(loss), "NaN loss detected"

    loss.backward()
    for name, p in gla.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"
    optimizer.step()


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA package not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gla_equivalence(B, T, H, num_heads, dtype):
    """Compare our GLA to FLA's GatedLinearAttention in recurrent mode."""
    device = torch.device("cuda")
    x = torch.randn(B, T, H, dtype=dtype, device=device, requires_grad=True)

    model1 = (
        GatedLinearAttentionFLA(
            mode="fused_recurrent",
            hidden_size=H,
            num_heads=num_heads,
            expand_v=1,
            expand_k=1,
            fuse_norm=False,
            use_output_gate=False,
        )
        .to(dtype)
        .to(device)
    )
    model2 = (
        GatedLinearAttention(mode="recurrent", hidden_size=H, num_heads=num_heads)
        .to(dtype)
        .to(device)
    )
    model2.load_state_dict(model1.state_dict())

    o1, _, _ = model1(x)
    o2, _ = model2(x)

    assert torch.allclose(o1, o2, atol=1e-2)
