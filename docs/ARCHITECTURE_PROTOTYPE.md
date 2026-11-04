# Seto architecture prototype v0.1

Research prototype only. Existing 48K presets retain their dimensions, pre-norm
blocks, full causal attention, and tied embeddings. No post-norms or layer sharing
are added. The earlier research proposal's stability and memory estimates are
hypotheses, not results from this implementation.

## Selecting experiments

```python
from dataclasses import replace
from seto.config import MODEL_TINY

classic = replace(MODEL_TINY)
qk = replace(classic, qk_norm=True)
hybrid = replace(qk, sliding_window=256)
head128_only = replace(classic, head_dim=128)
# Alternative keeping Q projection width fixed: reduce Q and KV head counts.
head128_fixed_width = replace(classic, n_heads=8, n_kv_heads=2, head_dim=128)
```

`qk_norm` and `sliding_window` are independent; SWA without QK-Norm is also
supported. `attention_pattern` repeats across layers and defaults to
`("local", "local", "local", "global")`: zero-based layers 3, 7, ... are global.
Without `sliding_window`, every layer is global regardless of the pattern.
JSON lists are accepted for the pattern. Explicit `head_dim` decouples attention
projection width from residual width; keeping head counts while increasing it
increases attention parameters. Omitted or null `head_dim` remains `None` in the
configuration; `resolved_head_dim` derives the legacy quotient. Thus
`replace(classic, n_heads=8)` recomputes the implicit dimension, while an explicit
integer survives changes to head counts or residual width. Use
`replace(config, head_dim=None)` to return to automatic derivation. Both `asdict`
and `__dict__` JSON serialization preserve this distinction.

## QK normalization

Following the [Gemma 3 reference implementation](https://github.com/huggingface/transformers/blob/v4.52.2/src/transformers/models/gemma3/modeling_gemma3.py),
normalize each Q/K head independently over its last dimension before RoPE and KV
repetition. RMS and scale multiplication run in FP32, then cast back to the input
dtype. Each layer has separate Q and K scale vectors of length `head_dim`, shared
across heads and tokens, but never across layers or between Q and K. The stored
weights start at zero and multiply by `1 + weight`, initially a unit scale.
This adds exactly `2 * head_dim` parameters per layer, correcting the earlier
proposal's count of separate scales for every head. Existing block RMSNorm stays
unchanged. SDPA retains its default `1/sqrt(head_dim)` scaling. No FP16 stability
or model-quality improvement has been established by these CPU tests.

## Sliding window semantics and limits

For query position `i`, keep key `j` iff `0 <= i-j < window`. The window includes
the current token, so window 1 allows self only; window 256 allows up to 255
preceding tokens. Global layers attend to all preceding tokens and self. Local
layers pass a boolean keep mask with `is_causal=False`; global layers pass no mask
with `is_causal=True`. Both native SDPA GQA and repeated-KV fallback use these
same arguments.

The local mask is dense O(T²) storage and does **not** supply sparse-compute
acceleration. There is no bounded KV cache implementation or measured cache
saving. Multilayer receptive fields can extend beyond a single local window.

## Parameters and reproducibility

With residual width D, head dimension H, Q heads N, KV heads K, feed-forward width
F, vocabulary V and L layers:

- Embedding: V*D; untied output adds another V*D.
- Each layer: 2*D*H*(N+K) + 3*D*F + 2*D + (2*H if QK-Norm).
- Final norm: D. SWA adds no parameters.

Run `python -m scripts.architecture_report` for CSV counts of all three original
presets and four experiments. It constructs real module structures on PyTorch's
metadata-only device and counts unique parameter objects, checking against the
formula without allocating large weights. Unit tests also verify actual CPU
weight allocations for small tied/untied models, including head dimension 128.

## Compatibility and verification

Old JSON configuration dictionaries load with experiment defaults disabled.
Classic state-dict names and shapes stay unchanged, permitting strict legacy
loading. New JSON serialization includes nullable `head_dim` and experiment
fields; old software may reject these extra fields. Enabling QK-Norm adds scale
keys, so strict loading of a Classic checkpoint into that architecture fails as
expected. Previously serialized integer `head_dim` values still load with the same
projection shapes and are treated as explicit; their original implicit/explicit
intent cannot be recovered. A deliberate `strict=False` warm start retains zero-initialized scales
but changes model behavior. Changing head dimension changes projection shapes.

Run `python -m pytest tests/test_architecture.py -q`. Tests use tiny CPU models,
check forward/loss/gradients, causality, exact window boundaries, QK scales,
local/global patterns, MHA/GQA and fallback, counts, and JSON/state-dict round trips.

Verified counts (formula equals unique physical parameters):

| Preset | Classic | QK-Norm | QK-Norm + SWA | head128 only |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 206,992,384 | 206,994,176 | 206,994,176 | 243,692,544 |
| Small | 490,108,160 | 490,111,232 | 490,111,232 | 588,412,160 |
| Base | 1,073,047,552 | 1,073,053,184 | 1,073,053,184 | 1,073,047,552 |

Validation: 37 CPU tests passed with Python 3.12 / PyTorch 2.2.2, which exercises
the repeated-KV compatibility path because native SDPA GQA is unavailable there.
The native-GQA path remains dependent on the installed PyTorch backend; no GPU
runs were performed. Focused Ruff checks, syntax compilation, and diff whitespace
checks also pass.
