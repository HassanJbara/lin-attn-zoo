import pytest
import torch
from models.deltanet import DeltaNetConfig, DeltaNetModel


@pytest.fixture
def model_config():
    """Create a test configuration for DeltaNet."""
    return DeltaNetConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=4,
        conv_size=3,
        norm_eps=1e-5,
    )


@pytest.fixture
def model(model_config):
    """Create a DeltaNet model for testing."""
    model = DeltaNetModel(model_config)
    model.eval()
    return model


@pytest.fixture
def test_inputs(model_config):
    """Create test input tensors."""
    batch_size = 2
    seq_length = 10
    return {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "input_ids": torch.randint(
            0, model_config.vocab_size, (batch_size, seq_length)
        ),
    }


def test_forward_no_memory(model, test_inputs, model_config):
    """Test forward pass without providing memory states."""
    with torch.no_grad():
        logits, hidden_states, memory_states = model(input_ids=test_inputs["input_ids"])

    # Check shapes
    assert logits.shape == (
        test_inputs["batch_size"],
        test_inputs["seq_length"],
        model_config.vocab_size,
    )
    assert hidden_states.shape == (
        test_inputs["batch_size"],
        test_inputs["seq_length"],
        model_config.hidden_size,
    )
    assert len(memory_states) == model_config.num_hidden_layers

    # Check memory state shapes
    head_dim = model_config.hidden_size // model_config.num_heads
    for state in memory_states:
        assert state.shape == (
            test_inputs["batch_size"],
            model_config.num_heads,
            head_dim,
            head_dim,
        )


def test_forward_with_memory(model, test_inputs, model_config):
    """Test forward pass with provided memory states."""
    # Create initial memory states
    batch_size = test_inputs["batch_size"]
    head_dim = model_config.hidden_size // model_config.num_heads

    memory_states = [
        torch.zeros((batch_size, model_config.num_heads, head_dim, head_dim))
        for _ in range(model_config.num_hidden_layers)
    ]

    with torch.no_grad():
        logits, hidden_states, updated_memory_states = model(
            input_ids=test_inputs["input_ids"], memory_states=memory_states
        )

    # Check shapes
    assert logits.shape == (
        batch_size,
        test_inputs["seq_length"],
        model_config.vocab_size,
    )
    assert hidden_states.shape == (
        batch_size,
        test_inputs["seq_length"],
        model_config.hidden_size,
    )
    assert len(updated_memory_states) == model_config.num_hidden_layers

    # Verify memory states were updated (should not be all zeros anymore)
    for state in updated_memory_states:
        assert not torch.allclose(state, torch.zeros_like(state))


def test_autoregressive_generation(model, test_inputs, model_config):
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
            logits, _, memory_states = model(
                input_ids=current_input, memory_states=memory_states
            )

            # Verify logits shape for single token prediction
            assert logits.shape == (batch_size, 1, model_config.vocab_size)

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
        assert (tokens < model_config.vocab_size).all()
