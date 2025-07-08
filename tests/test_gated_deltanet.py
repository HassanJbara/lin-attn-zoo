import pytest
import torch
from models.gated_deltanet import GatedDeltaNet, GatedDeltaNetModel, GatedDeltaNetConfig

# Try to import FLA, but handle if not installed
try:
    from fla.layers import GatedDeltaNet as GatedDeltaNetFLA

    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False


# # global variables for testing
HIDDEN_SIZES = [64]
VOCAB_SIZE = 10000
NUM_HEADS = [3]
SEQ_LENGTHS = [256, 500]
BATCH_SIZES = [1, 3]
DTYPES = [torch.float16, torch.float32]


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_model_forward(B, T, mode, num_heads, dtype):
    """Test forward pass without providing memory states."""
    gated_model_config = GatedDeltaNetConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=num_heads,
        head_dim=32,
        mode=mode,
    )
    model = GatedDeltaNetModel(gated_model_config).to(dtype)
    input_ids = torch.randint(0, gated_model_config.vocab_size, (B, T))
    with torch.no_grad():
        logits, hidden_states, memory_states = model(input_ids=input_ids)

    # Check shapes
    assert logits.shape == (B, T, gated_model_config.vocab_size)
    assert hidden_states.shape == (B, T, gated_model_config.hidden_size)
    assert len(memory_states) == gated_model_config.num_hidden_layers

    # Check memory state shapes
    for state in memory_states:
        assert state.shape == (
            B,
            gated_model_config.num_heads,
            gated_model_config.head_dim,
            gated_model_config.head_dim,
        )


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_layer_modes(B: int, T: int, H: int, num_heads: int, dtype: torch.dtype):
    model1 = GatedDeltaNet(mode="recurrent", hidden_size=H, num_heads=num_heads).to(
        dtype
    )

    model2 = GatedDeltaNet(mode="chunk", hidden_size=H, num_heads=num_heads).to(dtype)

    model2.load_state_dict(model1.state_dict())
    model1.eval()
    model2.eval()

    x = torch.randn(B, T, H).to(dtype).requires_grad_(True)
    o1, _ = model1(x)
    o2, _ = model2(x)

    assert torch.allclose(o1, o2, atol=1e-2)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
@pytest.mark.parametrize(
    "dtype", [torch.float32]
)  # float16 is not that numerically stable
def test_layer_backpropagation(
    B: int, T: int, H: int, num_heads: int, mode: str, dtype: torch.dtype
):
    """Test backpropagation."""
    layer = GatedDeltaNet(mode=mode, hidden_size=H, num_heads=num_heads).to(dtype)

    # Switch to training mode
    layer.train()
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    optimizer.zero_grad()
    x = torch.randn(B, T, H).to(dtype)
    out, _ = layer(x)

    loss = out.sum()
    assert not torch.isnan(loss).any(), f"NaN loss detected for {mode} mode"

    loss.backward()
    for name, param in layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    optimizer.step()


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA package not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gated_delta_net_equivalence(
    B: int, T: int, H: int, mode: str, dtype: torch.dtype
):
    device = torch.device("cuda")
    x = torch.randn(B, T, H).to(dtype).to(device).requires_grad_(True)
    n_heads = 2

    model1 = (
        GatedDeltaNetFLA(
            mode="chunk",
            hidden_size=H,
            num_heads=n_heads,
            expand_v=1,
        )
        .to(dtype)
        .to(device)
    )
    model2 = GatedDeltaNet(
        hidden_size=H,
        num_heads=n_heads,
        conv_size=4,
        norm_eps=1e-5,
        mode=mode,
    ).to(device)
    model2.load_state_dict(model1.state_dict())

    o1, _ = model1(x)
    o2, _, _ = model2(x, use_cache=True)

    assert torch.allclose(o1, o2, atol=1e-2)
