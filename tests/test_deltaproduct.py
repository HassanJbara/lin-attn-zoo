import pytest
import torch

from models.deltaproduct import GatedDeltaProduct

try:
    from fla.layers import GatedDeltaProduct as GatedDeltaProductFLA

    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False


# Global variables for testing
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
def test_deltaproduct_forward_shapes(B, T, H, num_heads, dtype):
    """Test that forward pass produces correct output shapes."""
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent", hidden_size=H, num_heads=num_heads, head_dim=head_dim
    ).to(dtype)
    x = torch.randn(B, T, H, dtype=dtype)
    o, state = model(x)

    # Output should be [B, T, H]
    assert o.shape == (B, T, H)
    # State should be [B, num_heads, head_dim, head_dim]
    assert state.shape == (B, num_heads, head_dim, head_dim)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_gate", [True, False])
def test_deltaproduct_with_gate_options(B, T, H, num_heads, dtype, use_gate):
    """Test that the model works with and without output gating."""
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent",
        hidden_size=H,
        num_heads=num_heads,
        head_dim=head_dim,
        use_gate=use_gate,
    ).to(dtype)
    x = torch.randn(B, T, H, dtype=dtype)
    o, state = model(x)

    # Output should be [B, T, H]
    assert o.shape == (B, T, H)
    # State should be [B, num_heads, head_dim, head_dim]
    assert state.shape == (B, num_heads, head_dim, head_dim)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("allow_neg_eigval", [True, False])
def test_deltaproduct_neg_eigval_options(B, T, H, num_heads, dtype, allow_neg_eigval):
    """Test that the model works with different allow_neg_eigval settings."""
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent",
        hidden_size=H,
        num_heads=num_heads,
        head_dim=head_dim,
        allow_neg_eigval=allow_neg_eigval,
    ).to(dtype)
    x = torch.randn(B, T, H, dtype=dtype)
    o, state = model(x)

    assert o.shape == (B, T, H)
    assert state.shape == (B, num_heads, head_dim, head_dim)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_householder", [1, 2, 3])
def test_deltaproduct_householder_variations(B, T, H, num_heads, num_householder):
    """Test that the model works with different numbers of Householder reflections."""
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent",
        hidden_size=H,
        num_heads=num_heads,
        head_dim=head_dim,
        num_householder=num_householder,
    ).to(torch.float32)
    x = torch.randn(B, T, H, dtype=torch.float32)
    o, state = model(x)

    assert o.shape == (B, T, H)
    assert state.shape == (B, num_heads, head_dim, head_dim)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
def test_deltaproduct_backpropagation(B, T, H, num_heads):
    """Test that backpropagation works and produces valid gradients."""
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent", hidden_size=H, num_heads=num_heads, head_dim=head_dim
    ).to(torch.float32)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    x = torch.randn(B, T, H, dtype=torch.float32)
    out, state = model(x)
    loss = out.sum()
    assert not torch.isnan(loss), "NaN loss detected in recurrent mode"

    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name} in recurrent mode"
            assert not torch.isnan(p.grad).any(), (
                f"NaN grad for {name} in recurrent mode"
            )
    optimizer.step()


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
def test_deltaproduct_recurrent_state_persistence(B, H, num_heads):
    """Test that recurrent state is properly maintained across forward passes."""
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent", hidden_size=H, num_heads=num_heads, head_dim=head_dim
    ).to(torch.float32)
    model.eval()

    # First sequence
    x1 = torch.randn(B, 8, H, dtype=torch.float32)
    o1, state1 = model(x1)

    # Second sequence using the state from the first
    x2 = torch.randn(B, 8, H, dtype=torch.float32)
    o2, state2 = model(x2, recurrent_state=state1)

    # Check that states have evolved
    assert not torch.allclose(state1, state2)

    # Check that output shapes are correct
    assert o1.shape == (B, 8, H)
    assert o2.shape == (B, 8, H)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_deltaproduct_no_nans_or_infs(B, T, H, num_heads, dtype):
    """Test that the model doesn't produce NaN or Inf values."""
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent", hidden_size=H, num_heads=num_heads, head_dim=head_dim
    ).to(dtype)
    model.eval()

    x = torch.randn(B, T, H, dtype=dtype)
    o, state = model(x)

    assert not torch.isnan(o).any(), "NaN values in output"
    assert not torch.isinf(o).any(), "Inf values in output"
    assert not torch.isnan(state).any(), "NaN values in state"
    assert not torch.isinf(state).any(), "Inf values in state"


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
def test_deltaproduct_deterministic(B, H, num_heads):
    """Test that the model produces deterministic outputs for the same input."""
    torch.manual_seed(42)
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent", hidden_size=H, num_heads=num_heads, head_dim=head_dim
    ).to(torch.float32)
    model.eval()

    x = torch.randn(B, 16, H, dtype=torch.float32)

    # First forward pass
    o1, state1 = model(x)

    # Second forward pass with the same input
    o2, state2 = model(x)

    # Outputs should be identical
    assert torch.allclose(o1, o2, atol=1e-6)
    assert torch.allclose(state1, state2, atol=1e-6)


@pytest.mark.parametrize("conv_size", [2, 4, 8])
def test_deltaproduct_conv_sizes(conv_size):
    """Test that the model works with different convolution kernel sizes."""
    head_dim = 64 // 4  # 16
    model = GatedDeltaProduct(
        mode="recurrent",
        hidden_size=64,
        num_heads=4,
        head_dim=head_dim,
        conv_size=conv_size,
    ).to(torch.float32)

    x = torch.randn(2, 16, 64, dtype=torch.float32)
    o, state = model(x)

    assert o.shape == (2, 16, 64)
    assert state.shape == (2, 4, head_dim, head_dim)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_householder", [1, 2, 3])
def test_causality(B, T, H, dtype, num_heads, num_householder):
    """Test that the model maintains causality in its outputs."""
    head_dim = H // num_heads
    model = GatedDeltaProduct(
        mode="recurrent",
        hidden_size=H,
        num_heads=num_heads,
        head_dim=head_dim,
        num_householder=num_householder,
    ).to(dtype)
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
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
def test_fla_equivalence(B: int, T: int, H: int, dtype: torch.dtype, num_heads: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model1 = (
        GatedDeltaProduct(
            mode="recurrent",
            hidden_size=H,
            num_heads=num_heads,
            head_dim=H // num_heads,
            # chunk_size=64,
        )
        .to(dtype)
        .to(device)
    )

    model2 = (
        GatedDeltaProductFLA(
            mode="fused_recurrent",
            hidden_size=H,
            num_heads=num_heads,
            head_dim=H // num_heads,
            expand_v=1,
            # chunk_size=64,
        )
        .to(dtype)
        .to(device)
    )

    model2.load_state_dict(model1.state_dict())
    model1.eval()
    model2.eval()

    x = torch.randn(B, T, H).to(dtype).to(device).requires_grad_(True)

    o1, _ = model1(x)
    o2, _, _ = model2(x, use_cache=True)

    assert torch.allclose(o1, o2, atol=1e-2)
