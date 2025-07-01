import pytest
import torch
from models.gated_deltanet import GatedDeltaNet, GatedDeltaNetModel, GatedDeltaNetConfig

# Try to import FLA, but handle if not installed
try:
    from fla.layers import GatedDeltaNet as GatedDeltaNetFLA

    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False


# global variables for testing
HIDDEN_SIZE = 128
VOCAB_SIZE = 10000
NUM_HEADS = 3
SEQ_LENGTH = 1000
BATCH_SIZE = 2


@pytest.fixture
def gated_model_config():
    """Create a test configuration for GatedDeltaNet."""
    return GatedDeltaNetConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        expand_v=1,
        head_dim=32,
        num_heads=NUM_HEADS,
        mode="recurrent",
        use_gate=True,
        use_short_conv=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-5,
    )


@pytest.fixture
def gated_chunk_model_config():
    """Create a test configuration for GatedDeltaNet."""
    return GatedDeltaNetConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        expand_v=1,
        head_dim=32,
        num_heads=NUM_HEADS,
        mode="chunk",
        use_gate=True,
        use_short_conv=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-5,
    )


@pytest.fixture
def gated_model(gated_model_config):
    """Create a GatedDeltaNet model for testing."""
    model = GatedDeltaNetModel(gated_model_config)
    model.eval()
    return model


@pytest.fixture
def chunk_model(gated_chunk_model_config):
    """Create a DeltaNet model in chunk mode for testing."""
    model = GatedDeltaNetModel(gated_chunk_model_config)
    model.eval()
    return model


@pytest.fixture
def test_inputs():
    """Create test input tensors."""
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH)),
        "input_embeds": torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
    }


def test_forward(gated_model, test_inputs, gated_model_config):
    """Test forward pass without providing memory states."""
    with torch.no_grad():
        logits, hidden_states, memory_states = gated_model(
            input_ids=test_inputs["input_ids"]
        )

    # Check shapes
    assert logits.shape == (
        test_inputs["batch_size"],
        test_inputs["seq_length"],
        gated_model_config.vocab_size,
    )
    assert hidden_states.shape == (
        test_inputs["batch_size"],
        test_inputs["seq_length"],
        gated_model_config.hidden_size,
    )
    assert len(memory_states) == gated_model_config.num_hidden_layers

    # Check memory state shapes
    for state in memory_states:
        assert state.shape == (
            test_inputs["batch_size"],
            gated_model_config.num_heads,
            gated_model_config.head_dim,
            gated_model_config.head_dim,
        )


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("T", [256, 500])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("num_heads", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_modes(B: int, T: int, H: int, num_heads: int, dtype: torch.dtype):
    model1 = GatedDeltaNet(
        mode="recurrent", hidden_size=H, expand_v=1, num_heads=num_heads
    ).to(dtype)

    model2 = GatedDeltaNet(
        mode="chunk", hidden_size=H, expand_v=1, num_heads=num_heads
    ).to(dtype)

    model2.load_state_dict(model1.state_dict())
    model1.eval()
    model2.eval()

    x = torch.randn(B, T, H).to(dtype).requires_grad_(True)
    o1, _ = model1(x)
    o2, _ = model2(x)

    assert torch.allclose(o1, o2, atol=1e-2)


# def test_compare_modes_output(chunk_model, gated_model, test_inputs):
#     """Test that both modes produce outputs with the same values."""
#     # copy weights from chunk model to recurrent model
#     gated_model.load_state_dict(chunk_model.state_dict())

#     with torch.no_grad():
#         chunk_logits, chunk_hidden, chunk_memory = chunk_model(
#             input_ids=test_inputs["input_ids"]
#         )
#         recurrent_logits, recurrent_hidden, recurrent_memory = gated_model(
#             input_ids=test_inputs["input_ids"]
#         )

#     # Compare values
#     assert torch.allclose(chunk_logits, recurrent_logits, atol=1e-1)
#     assert torch.allclose(chunk_hidden, recurrent_hidden, atol=1e-2)

#     # Check memory states are not all zeros
#     for state in chunk_memory:
#         assert not torch.allclose(state, torch.zeros_like(state))
#     for state in recurrent_memory:
#         assert not torch.allclose(state, torch.zeros_like(state))


def test_autoregressive_generation(gated_model, test_inputs, gated_model_config):
    """Test autoregressive token generation."""
    batch_size = test_inputs["batch_size"]

    # Start with first token only
    current_input = test_inputs["input_ids"][:, 0:1]
    memory_states = None

    # Track generated tokens
    generated_tokens = []

    # Generate 5 tokens
    for _ in range(5):
        with torch.no_grad():
            logits, _, memory_states = gated_model(
                input_ids=current_input, memory_states=memory_states
            )

            # Verify logits shape for single token prediction
            assert logits.shape == (batch_size, 1, gated_model_config.vocab_size)

            # Get next token predictions
            next_tokens = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            assert next_tokens.shape == (batch_size, 1)

            # Use as next input
            current_input = next_tokens
            generated_tokens.append(next_tokens)

    # Verify we generated 5 tokens for each sequence
    assert len(generated_tokens) == 5

    # Verify all tokens are within vocabulary range
    for tokens in generated_tokens:
        assert (tokens >= 0).all()
        assert (tokens < gated_model_config.vocab_size).all()


def test_backpropagation(chunk_model, test_inputs, gated_model_config):
    """Test backpropagation."""
    # Switch to training mode
    chunk_model.train()
    optimizer = torch.optim.Adam(chunk_model.parameters(), lr=0.001)
    input_ids = test_inputs["input_ids"]
    target_ids = torch.randint(0, gated_model_config.vocab_size, input_ids.shape)
    optimizer.zero_grad()

    loss, _, _ = chunk_model(input_ids=input_ids, labels=target_ids)
    assert not torch.isnan(loss).any()

    loss.backward()
    for name, param in chunk_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    optimizer.step()


def test_backpropagation_no_gate(
    gated_model_no_gate, test_inputs, gated_model_config_no_gate
):
    """Test backpropagation with no gating."""
    # Switch to training mode
    gated_model_no_gate.train()
    optimizer = torch.optim.Adam(gated_model_no_gate.parameters(), lr=0.001)
    input_ids = test_inputs["input_ids"]
    target_ids = torch.randint(
        0, gated_model_config_no_gate.vocab_size, input_ids.shape
    )
    optimizer.zero_grad()

    loss, _, _ = gated_model_no_gate(input_ids=input_ids, labels=target_ids)
    assert not torch.isnan(loss).any()

    loss.backward()
    for name, param in gated_model_no_gate.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    optimizer.step()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_length", [1, 10])
def test_different_input_shapes(
    batch_size, seq_length, gated_model, gated_model_config
):
    """Test model with different input shapes."""
    input_ids = torch.randint(
        0, gated_model_config.vocab_size, (batch_size, seq_length)
    )

    with torch.no_grad():
        logits, hidden_states, memory_states = gated_model(input_ids=input_ids)

    # Check shapes
    assert logits.shape == (batch_size, seq_length, gated_model_config.vocab_size)
    assert hidden_states.shape == (
        batch_size,
        seq_length,
        gated_model_config.hidden_size,
    )
    assert len(memory_states) == gated_model_config.num_hidden_layers


def test_gated_vs_nongated_output_shapes(gated_model, gated_model_no_gate, test_inputs):
    """Test that both gated and non-gated models produce outputs with the same shapes."""
    with torch.no_grad():
        gated_logits, gated_hidden, gated_memory = gated_model(
            input_ids=test_inputs["input_ids"]
        )
        nongated_logits, nongated_hidden, nongated_memory = gated_model_no_gate(
            input_ids=test_inputs["input_ids"]
        )

    # Compare shapes
    assert gated_logits.shape == nongated_logits.shape
    assert gated_hidden.shape == nongated_hidden.shape
    assert len(gated_memory) == len(nongated_memory)


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA package not installed")
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gated_delta_net_equivalence(
    B: int, T: int, H: int, dtype: torch.dtype, mode: str
):
    torch.manual_seed(47)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(B, T, H).to(dtype).to(device).requires_grad_(True)
    n_heads = 2

    model1 = (
        GatedDeltaNetFLA(
            mode="fused_recurrent",
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
        mode="recurrent",
        expand_v=1,
    ).to(device)
    model2.load_state_dict(model1.state_dict())

    o1, _ = model1(x)
    o2, _, _ = model2(x, use_cache=True)

    assert torch.allclose(o1, o2, atol=1e-2)
