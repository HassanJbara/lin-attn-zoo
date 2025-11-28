import pytest
import torch

from models.gla import GatedLinearAttention

try:
    from fla.layers import GatedLinearAttention as GatedLinearAttentionFLA

    FLA_AVAILABLE = True
except Exception:
    FLA_AVAILABLE = False


# global variables for testing
BATCH_SIZES = [1, 3]
SEQ_LENGTHS = [16, 32]
HIDDEN_SIZES = [64]
NUM_HEADS = [4]
DTYPES = [torch.float32, torch.float16]
MODES = ["recurrent", "chunk"]


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_gla_forward_shapes(B, T, H, num_heads, dtype, mode):
    gla = GatedLinearAttention(mode=mode, hidden_size=H, num_heads=num_heads).to(dtype)
    x = torch.randn(B, T, H, dtype=dtype)
    o, state = gla(x)

    # output should be [B, T, H]
    assert o.shape == (B, T, H)
    head_dim = H // num_heads
    # state should be [B, num_heads, head_dim, head_dim]
    assert state.shape == (B, num_heads, head_dim, head_dim)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gla_mode_equivalence(B, T, H, num_heads, dtype):
    """Test for numerical equivalence between recurrent and chunk modes."""
    recurrent_model = GatedLinearAttention(
        mode="recurrent", hidden_size=H, num_heads=num_heads
    ).to(dtype)
    chunk_model = GatedLinearAttention(
        mode="chunk", hidden_size=H, num_heads=num_heads, chunk_size=16
    ).to(dtype)

    chunk_model.load_state_dict(recurrent_model.state_dict())
    recurrent_model.eval()
    chunk_model.eval()

    x = torch.randn(B, T, H, dtype=dtype)
    o_recurrent, _ = recurrent_model(x)
    o_chunk, _ = chunk_model(x)

    assert torch.allclose(o_recurrent, o_chunk, atol=1e-2)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("mode", MODES)
def test_gla_backpropagation(B, T, H, num_heads, mode):
    gla = GatedLinearAttention(
        mode=mode, hidden_size=H, num_heads=num_heads, chunk_size=16
    ).to(torch.float32)
    gla.train()
    optimizer = torch.optim.Adam(gla.parameters(), lr=1e-3)
    optimizer.zero_grad()

    x = torch.randn(B, T, H, dtype=torch.float32)
    out, state = gla(x)
    loss = out.sum()
    assert not torch.isnan(loss), f"NaN loss detected in {mode} mode"

    loss.backward()
    for name, p in gla.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name} in {mode} mode"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name} in {mode} mode"
    optimizer.step()


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("mode", MODES)
def test_causality(B, T, H, dtype, num_heads, mode):
    """Test that the model maintains causality in its outputs."""
    model = GatedLinearAttention(mode=mode, hidden_size=H, num_heads=num_heads).to(
        dtype
    )
    model.eval()

    x_a = torch.randn(B, T, H, dtype=dtype)
    x_b = x_a.clone()
    x_b[:, T:] = torch.randn(B, T - T, H, dtype=dtype)  # different future

    with torch.no_grad():
        y_a, _ = model(x_a)
        y_b, _ = model(x_b)

    assert torch.allclose(y_a[:, :T], y_b[:, :T], atol=1e-5), "Causality violated"


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA package not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_output_gate", [True, False])
@pytest.mark.parametrize("mode", MODES)
def test_gla_equivalence(B, T, H, num_heads, dtype, use_output_gate, mode):
    """Compare our GLA to FLA's GatedLinearAttention."""
    device = torch.device("cuda")
    x = torch.randn(B, T, H, dtype=dtype, device=device, requires_grad=True)

    fla_mode = "fused_recurrent" if mode == "recurrent" else "fused_chunk"

    model1 = (
        GatedLinearAttentionFLA(
            mode=fla_mode,
            hidden_size=H,
            num_heads=num_heads,
            use_output_gate=use_output_gate,
            chunk_size=16,
        )
        .to(dtype)
        .to(device)
    )
    model2 = (
        GatedLinearAttention(
            mode=mode,
            hidden_size=H,
            num_heads=num_heads,
            use_output_gate=use_output_gate,
            chunk_size=16,
        )
        .to(dtype)
        .to(device)
    )
    model2.load_state_dict(model1.state_dict())

    o1, _ = model1(x)
    o2, _ = model2(x)

    assert torch.allclose(o1, o2, atol=1e-2)
