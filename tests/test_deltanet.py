import pytest
import torch
from models.deltanet import DeltaNet, DeltaNetConfig, DeltaNetModel

# Try to import FLA, but handle if not installed
try:
    from fla.layers import DeltaNet as DeltaNetFLA

    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False


@pytest.fixture
def chunk_model_config():
    """Create a test configuration for DeltaNet in chunk mode."""
    return DeltaNetConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=4,
        conv_size=3,
        norm_eps=1e-5,
        mode="chunk",
        chunk_size=4,
    )


@pytest.fixture
def recurrent_model_config():
    """Create a test configuration for DeltaNet in recurrent mode."""
    return DeltaNetConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=4,
        conv_size=3,
        norm_eps=1e-5,
        mode="recurrent",
    )


@pytest.fixture
def chunk_model(chunk_model_config):
    """Create a DeltaNet model in chunk mode for testing."""
    model = DeltaNetModel(chunk_model_config)
    model.eval()
    return model


@pytest.fixture
def recurrent_model(recurrent_model_config):
    """Create a DeltaNet model in recurrent mode for testing."""
    model = DeltaNetModel(recurrent_model_config)
    model.eval()
    return model


@pytest.fixture
def test_inputs():
    """Create test input tensors."""
    batch_size = 2
    seq_length = 10
    vocab_size = 10000
    return {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
    }


def test_forward_no_memory(chunk_model, test_inputs, chunk_model_config):
    """Test forward pass without providing memory states for chunk mode."""
    with torch.no_grad():
        logits, hidden_states, memory_states = chunk_model(
            input_ids=test_inputs["input_ids"]
        )

    # Check shapes
    assert logits.shape == (
        test_inputs["batch_size"],
        test_inputs["seq_length"],
        chunk_model_config.vocab_size,
    )
    assert hidden_states.shape == (
        test_inputs["batch_size"],
        test_inputs["seq_length"],
        chunk_model_config.hidden_size,
    )
    assert len(memory_states) == chunk_model_config.num_hidden_layers

    # Check memory state shapes - updated for the new format [B, 1, H, D, D]
    head_dim = chunk_model_config.hidden_size // chunk_model_config.num_heads
    for state in memory_states:
        assert state.shape == (
            test_inputs["batch_size"],
            1,  # New dimension for the state
            chunk_model_config.num_heads,
            head_dim,
            head_dim,
        )


def test_forward_recurrent_mode(recurrent_model, test_inputs, recurrent_model_config):
    """Test forward pass in recurrent mode."""
    with torch.no_grad():
        logits, hidden_states, memory_states = recurrent_model(
            input_ids=test_inputs["input_ids"]
        )

    # Check shapes
    assert logits.shape == (
        test_inputs["batch_size"],
        test_inputs["seq_length"],
        recurrent_model_config.vocab_size,
    )
    assert hidden_states.shape == (
        test_inputs["batch_size"],
        test_inputs["seq_length"],
        recurrent_model_config.hidden_size,
    )
    assert len(memory_states) == recurrent_model_config.num_hidden_layers


def test_forward_with_memory_chunk_mode(chunk_model, test_inputs, chunk_model_config):
    """Test forward pass with provided memory states in chunk mode."""
    # Create initial memory states - updated shape to [B, 1, H, D, D]
    batch_size = test_inputs["batch_size"]
    head_dim = chunk_model_config.hidden_size // chunk_model_config.num_heads

    memory_states = [
        torch.zeros(
            (
                batch_size,
                1,  # Add dimension for state
                chunk_model_config.num_heads,
                head_dim,
                head_dim,
            )
        )
        for _ in range(chunk_model_config.num_hidden_layers)
    ]

    with torch.no_grad():
        logits, hidden_states, updated_memory_states = chunk_model(
            input_ids=test_inputs["input_ids"], memory_states=memory_states
        )

    # Check shapes
    assert logits.shape == (
        batch_size,
        test_inputs["seq_length"],
        chunk_model_config.vocab_size,
    )
    assert hidden_states.shape == (
        batch_size,
        test_inputs["seq_length"],
        chunk_model_config.hidden_size,
    )
    assert len(updated_memory_states) == chunk_model_config.num_hidden_layers

    # Verify memory states were updated (should not be all zeros anymore)
    for state in updated_memory_states:
        assert not torch.allclose(state, torch.zeros_like(state))
        # Check updated shape matches expected
        assert state.shape == (
            batch_size,
            1,
            chunk_model_config.num_heads,
            head_dim,
            head_dim,
        )


def test_autoregressive_generation_chunk_mode(
    chunk_model, test_inputs, chunk_model_config
):
    """Test autoregressive token generation in chunk mode."""
    batch_size = test_inputs["batch_size"]

    # Start with first token only
    current_input = test_inputs["input_ids"][:, 0:1]
    memory_states = None

    # Track generated tokens
    generated_tokens = []

    # Generate 5 tokens
    for _ in range(5):
        with torch.no_grad():
            logits, _, memory_states = chunk_model(
                input_ids=current_input, memory_states=memory_states
            )

            # Verify logits shape for single token prediction
            assert logits.shape == (batch_size, 1, chunk_model_config.vocab_size)

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
        assert (tokens < chunk_model_config.vocab_size).all()


def test_autoregressive_generation_recurrent_mode(
    recurrent_model, test_inputs, recurrent_model_config
):
    """Test autoregressive token generation in recurrent mode."""
    batch_size = test_inputs["batch_size"]

    # Start with first token only
    current_input = test_inputs["input_ids"][:, 0:1]
    memory_states = None

    # Track generated tokens
    generated_tokens = []

    # Generate 5 tokens
    for _ in range(5):
        with torch.no_grad():
            logits, _, memory_states = recurrent_model(
                input_ids=current_input, memory_states=memory_states
            )

            # Verify logits shape for single token prediction
            assert logits.shape == (batch_size, 1, recurrent_model_config.vocab_size)

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
        assert (tokens < recurrent_model_config.vocab_size).all()


def test_compare_modes_output_shapes(chunk_model, recurrent_model, test_inputs):
    """Test that both modes produce outputs with the same shapes."""
    with torch.no_grad():
        chunk_logits, chunk_hidden, chunk_memory = chunk_model(
            input_ids=test_inputs["input_ids"]
        )
        recurrent_logits, recurrent_hidden, recurrent_memory = recurrent_model(
            input_ids=test_inputs["input_ids"]
        )

    # Compare shapes
    assert chunk_logits.shape == recurrent_logits.shape
    assert chunk_hidden.shape == recurrent_hidden.shape
    assert len(chunk_memory) == len(recurrent_memory)


def test_compare_modes_output(chunk_model, recurrent_model, test_inputs):
    """Test that both modes produce outputs with the same values."""
    # copy weights from chunk model to recurrent model
    recurrent_model.load_state_dict(chunk_model.state_dict())
    recurrent_model.eval()
    chunk_model.eval()

    with torch.no_grad():
        chunk_logits, chunk_hidden, chunk_memory = chunk_model(
            input_ids=test_inputs["input_ids"]
        )
        recurrent_logits, recurrent_hidden, recurrent_memory = recurrent_model(
            input_ids=test_inputs["input_ids"]
        )

    # Compare values
    assert torch.allclose(chunk_logits, recurrent_logits, atol=1e-5)
    assert torch.allclose(chunk_hidden, recurrent_hidden, atol=1e-5)

    # Check memory states are not all zeros
    for state in chunk_memory:
        assert not torch.allclose(state, torch.zeros_like(state))
    for state in recurrent_memory:
        assert not torch.allclose(state, torch.zeros_like(state))


def test_backpropagation_chunk_mode(chunk_model, test_inputs, chunk_model_config):
    """Test backpropagation in chunk mode."""
    # Switch to training mode
    chunk_model.train()
    chunk_model = torch.compile(chunk_model, mode="reduce-overhead")
    optimizer = torch.optim.Adam(chunk_model.parameters(), lr=0.001)
    input_ids = test_inputs["input_ids"]
    target_ids = torch.randint(0, chunk_model_config.vocab_size, input_ids.shape)
    optimizer.zero_grad()

    loss, _, _ = chunk_model(input_ids=input_ids, labels=target_ids)
    assert not torch.isnan(loss).any()

    loss.backward()
    for name, param in chunk_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name} in chunk mode"
            assert not torch.isnan(param.grad).any(), (
                f"NaN gradient for {name} in chunk mode"
            )
    optimizer.step()


def test_backpropagation_recurrent_mode(
    recurrent_model, test_inputs, recurrent_model_config
):
    """Test backpropagation in recurrent mode."""
    # Switch to training mode
    recurrent_model.train()
    optimizer = torch.optim.Adam(recurrent_model.parameters(), lr=0.001)
    input_ids = test_inputs["input_ids"]
    target_ids = torch.randint(0, recurrent_model_config.vocab_size, input_ids.shape)
    optimizer.zero_grad()

    loss, _, _ = recurrent_model(input_ids=input_ids, labels=target_ids)
    assert not torch.isnan(loss).any()

    loss.backward()
    for name, param in recurrent_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name} in recurrent mode"
            assert not torch.isnan(param.grad).any(), (
                f"NaN gradient for {name} in recurrent mode"
            )
    optimizer.step()


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA package not installed")
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_delta_net_equivalence(B: int, T: int, H: int, dtype: torch.dtype, mode: str):
    torch.manual_seed(47)
    x = torch.randn(B, T, H).to(dtype).requires_grad_(True)
    n_heads = 2

    model1 = DeltaNet(
        hidden_size=H,
        num_heads=n_heads,
        conv_size=4,
        norm_eps=1e-5,
        mode=mode,
    ).to(dtype)

    model2 = DeltaNetFLA(
        mode=f"fused_'{mode}" if mode == "recurrent" else mode,
        d_model=H,
        num_heads=n_heads,
    ).to(dtype)

    model2.load_state_dict(model1.state_dict())

    o1, _ = model1(x)
    o2, _, _ = model2(x, use_cache=True)

    assert torch.allclose(o1, o2, atol=1e-2)
