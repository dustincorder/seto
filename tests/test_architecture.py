import io
import json
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from seto.config import MODEL_BASE, MODEL_SMALL, MODEL_TINY, ModelConfig
from seto.model import QKRMSNorm, SetoLM, sliding_window_mask


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(**kwargs):
    args = dict(
        vocab_size=19,
        d_model=16,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=24,
        max_seq_len=8,
        use_gradient_checkpointing=False,
    )
    return ModelConfig(**(args | kwargs))


@pytest.mark.parametrize("qk,window", [(False, None), (True, None), (True, 3), (False, 3)])
@pytest.mark.parametrize("kv", [1, 2, 4])
def test_forward_backward_counts(qk, window, kv):
    torch.manual_seed(42)
    config = tiny(qk_norm=qk, sliding_window=window, n_kv_heads=kv)
    model = SetoLM(config)
    ids = torch.randint(0, 19, (1, 6))
    logits, loss = model(ids, ids, return_logits=True)
    assert logits.shape == (1, 6, 19)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert config.num_params() == model.count_parameters()
    model.eval()
    changed = ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 19
    torch.testing.assert_close(model(ids)[0][:, :4], model(changed)[0][:, :4])


def test_mask_boundaries_and_sdpa():
    for length in (1, 6):
        for window in (1, 3, 10):
            mask = sliding_window_mask(length, window)
            for i in range(length):
                for j in range(length):
                    assert mask[i, j].item() == (max(0, i - window + 1) <= j <= i)
            q = torch.zeros(1, 1, length, 2)
            values = torch.arange(length).float().view(1, 1, length, 1)
            actual = F.scaled_dot_product_attention(q, q, values, attn_mask=mask)
            expected = torch.tensor(
                [
                    sum(range(max(0, i - window + 1), i + 1)) / min(i + 1, window)
                    for i in range(length)
                ]
            )
            torch.testing.assert_close(actual.flatten(), expected)


def test_norm_and_pattern():
    norm = QKRMSNorm(4, 1e-6)
    assert norm.weight.shape == (4,)
    assert torch.count_nonzero(norm.weight) == 0
    x = torch.randn(2, 3, 5, 4)
    torch.testing.assert_close(norm(x), x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6))
    model = SetoLM(tiny(qk_norm=True, sliding_window=3, n_layers=8))
    assert [layer.attention.window for layer in model.layers] == [3, 3, 3, None] * 2
    assert model.layers[0].attention.q_norm is not model.layers[0].attention.k_norm
    custom = SetoLM(tiny(sliding_window=2, attention_pattern=("global", "local")))
    assert [layer.attention.window for layer in custom.layers] == [None, 2, None, 2]


@pytest.mark.parametrize("tied", [True, False])
def test_head_dim_serialization_and_physical_count(tied):
    config = tiny(head_dim=128, qk_norm=True, sliding_window=3, tie_embeddings=tied)
    model = SetoLM(config)
    assert model(torch.zeros(1, 2, dtype=torch.long))[0].shape == (1, 2, 19)
    assert config.num_params() == sum(p.numel() for p in model.parameters())
    restored = SetoLM(ModelConfig(**json.loads(json.dumps(asdict(config)))))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert restored.config == config


def test_legacy_defaults():
    legacy = dict(
        vocab_size=19, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=24, max_seq_len=8
    )
    config = ModelConfig(**legacy)
    model = SetoLM(config)
    assert config.head_dim == 4 and not config.qk_norm and config.sliding_window is None
    assert not any("q_norm" in key or "k_norm" in key for key in model.state_dict())
    SetoLM(ModelConfig(**legacy)).load_state_dict(model.state_dict(), strict=True)
    # Verify full presets without allocating their storage.
    for preset in (MODEL_TINY, MODEL_SMALL, MODEL_BASE):
        assert preset.vocab_size == 48000
        with torch.device("meta"):
            physical = SetoLM(preset)
        assert physical.count_parameters() == preset.num_params()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_heads=0),
        dict(n_kv_heads=3),
        dict(d_model=15),
        dict(head_dim=3),
        dict(head_dim=0),
        dict(sliding_window=0),
        dict(attention_pattern=()),
        dict(attention_pattern=("bad",)),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        tiny(**kwargs)


def test_gqa_fallback_and_mask_contract():
    model = SetoLM(tiny(qk_norm=True, sliding_window=2))
    ids = torch.zeros(1, 5, dtype=torch.long)
    expected = model(ids)[0]
    original = F.scaled_dot_product_attention

    def fallback(*args, **kwargs):
        assert not (kwargs.get("attn_mask") is not None and kwargs.get("is_causal"))
        if kwargs.pop("enable_gqa", False):
            raise TypeError("simulate older torch")
        return original(*args, **kwargs)

    with patch.object(F, "scaled_dot_product_attention", side_effect=fallback):
        torch.testing.assert_close(model(ids)[0], expected)


def test_checkpointed_chunked_loss():
    model = SetoLM(tiny(qk_norm=True, sliding_window=2, use_gradient_checkpointing=True))
    ids = torch.randint(0, 19, (1, 5))
    _, loss = model(ids, ids)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
