# Architecture

> [!NOTE]
> Seto's architecture is under active research and reconsideration. The specifications outlined below reflect current experimental presets and prototypes, not a finalized production architecture.

Seto is designed as an efficient, compact language model optimized for on-device inference (laptops, smartphones, embedded hardware) under strict compute and memory budgets.

## Current Experimental Specifications

| Parameter | Tiny (~200M) | Small (~500M) | Base / 1B (~1.1B) |
| :--- | :--- | :--- | :--- |
| `d_model` | 1024 | 1280 | 2048 |
| Layers (`n_layers`) | 14 | 24 | 22 |
| Query Heads (`n_heads`) | 16 | 20 | 16 |
| Key/Value Heads (`n_kv_heads`) | 4 | 5 | 4 |
| FFN Dimension (`d_ff`) | 2816 (SwiGLU) | 3584 (SwiGLU) | 5504 (SwiGLU) |
| Context Length (`max_seq_len`) | 1024 | 2048 | 2048 |
| Vocabulary Size (`vocab_size`) | 48,000 | 48,000 | 48,000 |
| Normalization | RMSNorm (`eps=1e-6`) | RMSNorm (`eps=1e-6`) | RMSNorm (`eps=1e-6`) |
| Position Encoding | RoPE (`theta=10000.0`) | RoPE (`theta=10000.0`) | RoPE (`theta=10000.0`) |
| Attention Mechanism | GQA + SDPA | GQA + SDPA | GQA + SDPA |
| Embedding Tying | Tied (`lm_head = embed`) | Tied (`lm_head = embed`) | Tied (`lm_head = embed`) |

## Core Architectural Components

### 1. Grouped-Query Attention (GQA) & SDPA
- **GQA Ratio**: 4:1 query-to-KV head ratio across configurations to significantly reduce key-value cache memory during generation on low-RAM devices.
- **PyTorch SDPA**: Utilizes PyTorch's native Scaled Dot-Product Attention (`F.scaled_dot_product_attention`) with automatic dispatch (FlashAttention, Memory-Efficient, or Math kernels depending on runtime device capabilities).

### 2. Rotary Position Embeddings (RoPE)
- Rotary positional embeddings applied to queries and keys per layer to maintain relative token distance awareness.

### 3. SwiGLU Feed-Forward Network
- Gated Linear Units with SiLU activation:
  $$\text{FFN}(x) = W_{\text{down}} \left( \text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x \right)$$
- Implementation in `seto/model.py` (`SwiGLU`):
  ```python
  self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
  ```
- Scaled hidden dimension (`d_ff`) to balance transformer capacity without unnecessary parameter bloat.

### 4. RMSNorm & Pre-Normalization
- Pre-layer Root Mean Square Normalization applied before self-attention and FFN sub-layers, with an additional final normalization before the language modeling head.

### 5. Weight Tying
- Shared weights between input token embeddings and the output projection layer to minimize parameter count and memory consumption.

## Training & Memory Optimizations

To support training and experimentation on accessible hardware (e.g., dual Kaggle T4 GPUs):
- **Precision**: Native FP16 mixed precision support for NVIDIA Turing architecture (where BF16 is unavailable).
- **Gradient Checkpointing**: Configurable activation checkpointing to fit deeper variants into constrained VRAM.
- **Chunked Cross-Entropy**: Avoids allocating full logit tensors across large vocabulary slices during loss computation.
- **8-Bit AdamW**: Optional PagedAdamW8bit integration via `bitsandbytes` to reduce optimizer state footprint from ~8.6GB to ~2.1GB on ~1B models.

## Future Exploration

Architectural avenues under active evaluation:
- Dynamic context window extension and alternative RoPE scalings.
- Hardware-specific kernel quantization (4-bit / 8-bit integer quantization, GGUF/llama.cpp compatibility).
- Native tool-calling token schemas and structured function generation heads.
