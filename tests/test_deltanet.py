import pytest
import torch
from models.deltanet import DeltaNet
from models.model import ModelConfig, ForCausalLM

try:
    from fla.layers import DeltaNet as DeltaNetFLA

    FLA_AVAILABLE = True
except Exception:
    FLA_AVAILABLE = False


# Global variables for testing
HIDDEN_SIZES = [64]
VOCAB_SIZE = 10000
NUM_HEADS = [2, 4]
SEQ_LENGTHS = [256, 500]
BATCH_SIZES = [1, 3]
DTYPES = [torch.float16, torch.float32]
MODES = ["chunk", "recurrent"]


@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_model_forward(H, B, T, mode, num_heads, dtype):
    """Test forward pass without providing memory states."""
    model_config = ModelConfig(
        vocab_size=10000,
        hidden_size=H,
        num_hidden_layers=2,
        num_heads=num_heads,
        mode=mode,
    )
    model = ForCausalLM(model_config).to(dtype)
    input_ids = torch.randint(0, model_config.vocab_size, (B, T))
    with torch.no_grad():
        logits, hidden_states, memory_states = model(
            input_ids=input_ids, return_dict=False
        )

    # Check shapes
    assert logits.shape == (B, T, model_config.vocab_size)
    assert hidden_states.shape == (B, T, model_config.hidden_size)
    assert len(memory_states) == model_config.num_hidden_layers

    # Check memory state shapes
    for state in memory_states:
        if model_config.mode == "recurrent":
            assert state.shape == (
                B,
                1,
                model_config.num_heads,
                model_config.hidden_size // model_config.num_heads,
                model_config.hidden_size // model_config.num_heads,
            )
        else:
            assert state.shape == (
                B,
                model_config.num_heads,
                model_config.hidden_size // model_config.num_heads,
                model_config.hidden_size // model_config.num_heads,
            )


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_layer_modes(B: int, T: int, H: int, num_heads: int, dtype: torch.dtype):
    """Test that recurrent and chunk modes produce the same output."""
    model1 = DeltaNet(mode="recurrent", hidden_size=H, num_heads=num_heads).to(dtype)
    model2 = DeltaNet(mode="chunk", hidden_size=H, num_heads=num_heads).to(dtype)

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
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "dtype", [torch.float32]
)  # float16 is not that numerically stable
def test_layer_backpropagation(
    B: int, T: int, H: int, num_heads: int, mode: str, dtype: torch.dtype
):
    """Test backpropagation."""
    layer = DeltaNet(mode=mode, hidden_size=H, num_heads=num_heads).to(dtype)

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


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("mode", MODES)
def test_causality(B, T, H, dtype, num_heads, mode):
    """Test that the model is causal in both modes."""
    model = DeltaNet(mode=mode, hidden_size=H, num_heads=num_heads).to(dtype)
    model.eval()

    x_a = torch.randn(B, T + 10, H).to(dtype)
    x_b = x_a.clone()

    with torch.no_grad():
        y_a, _ = model(x_a)
        y_b, _ = model(x_b)

    assert torch.allclose(y_a[:, :T], y_b[:, :T], atol=1e-5), "Causality violated"


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA package not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("T", SEQ_LENGTHS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_delta_net_equivalence(B: int, T: int, H: int, mode: str, dtype: torch.dtype):
    """Test equivalence with FLA implementation."""
    device = torch.device("cuda")
    x = torch.randn(B, T, H).to(dtype).to(device).requires_grad_(True)
    n_heads = 2

    model1 = (
        DeltaNetFLA(
            mode="chunk",
            d_model=H,
            num_heads=n_heads,
        )
        .to(dtype)
        .to(device)
    )
    model2 = (
        DeltaNet(
            hidden_size=H,
            num_heads=n_heads,
            conv_size=4,
            norm_eps=1e-5,
            mode=mode,
        )
        .to(dtype)
        .to(device)
    )
    model2.load_state_dict(model1.state_dict())

    o1, _ = model1(x)
    o2, _, _ = model2(x, use_cache=True)

    assert torch.allclose(o1, o2, atol=1e-2)
