# Dataset Provenance and SFT Preparation

This document records data provenance specifications, curation workflows, and the SFT smoke-testing protocol for Seto.

---

## 1. Data Provenance & Sources

### Pretraining Corpus
- **FineWeb2 (`rus_Cyrl`)**: Primary Russian pretraining corpus (~85% target share).
- **Wikipedia (`20231101.ru`)**: Encyclopedic Russian text (~15% target share).
- Multilingual expansion targets: English and Ukrainian corpora to align with tokenizer capacity.

### Tokenizer
- **48,000 BPE vocabulary** trained across RU (80%), UK (10%), and EN (10%).
- **Special tokens**:
  - Boundary: `<bos>`, `<eos>`, `<pad>`
  - Roles: `<|system|>`, `<|user|>`, `<|assistant|>`
  - Tool calling: `<|tool_call|>`, `<|tool_result|>`

### SFT Corpus & Weighting
SFT records adhere to the canonical schema:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Target mix distribution (weighted across trainable assistant and tool-call turns):
- **40%** UltraChat instructions
- **15%** Cleaned WildChat casual dialogue
- **14%** Reviewed OpenAssistant (OASST1) human dialogue (RU/EN/UK)
- **11%** Cleaned WildChat multi-turn
- **10%** Russian instructions
- **10%** Hermes function calling / tool use

Source revisions, license metadata, rejection counts, and checksums are recorded alongside final artifacts. Validation dialogue is segregated and never sampled from OASST training trees (see `datasets/validation/README.md`).

---

## 2. Dataset Processing Pipeline

### OASST Extraction
Convert pinned OpenAssistant trees into deterministic, reviewed paths:

```bash
python scripts/prepare_oasst.py \
  --languages ru,en \
  --max-turns 12 \
  --max-chars 12000 \
  --min-reviews 1 \
  --output datasets/sft/oasst1.jsonl \
  --metadata-output datasets/sft/oasst1.meta.jsonl \
  --manifest-output datasets/sft/oasst1.manifest.json
```

### Filtering & Refusal Preservation
Filter general SFT conversational data while retaining bounded, safe refusals. Route roleplay conversations into designated subsets:

```bash
python scripts/sft_filter.py \
  --input datasets/sft/raw.jsonl \
  --output datasets/sft/clean.jsonl \
  --roleplay-output datasets/sft/roleplay.jsonl \
  --stats
```

### Multi-Source Mixing
Compile the weighted corpus based on configuration quotas without sample repetition:

```bash
python scripts/mix_sft.py \
  --config configs/sft-v1.json \
  --output datasets/sft/seto-sft-v1.jsonl \
  --metadata-output datasets/sft/seto-sft-v1.meta.jsonl \
  --manifest-output datasets/sft/seto-sft-v1.manifest.json
```

---

## 3. SFT Smoke-Test Protocol

Before launching full-scale alignment runs on GPU clusters or Kaggle/Colab environments, verify batching and gradient calculations using the included fixture `datasets/test-sft.jsonl`.

The fixture contains multi-turn conversations and a complete tool call, result, and final answer sequence.

### Step 1: Preflight Batch Check

```bash
python scripts/check_sft_batch.py \
  --dataset datasets/test-sft.jsonl \
  --tokenizer seto-tokenizer \
  --seq-len 1024
```

*Expected output*: Exactly 15 SFT targets (14 assistant turns and 1 tool call).

### Step 2: Smoke Training Run

```bash
python scripts/train.py \
  --stage sft \
  --model-config small \
  --dataset datasets/test-sft.jsonl \
  --tokenizer seto-tokenizer \
  --init-from seto-small/final_pretrain.zip \
  --output-dir smoke-sft \
  --max-steps 10 \
  --save-every 10
```

*Verification criteria*:
1. Loads previous checkpoint without shape mismatches.
2. Automatically resizes embedding matrix if vocabulary was extended.
3. Successfully masks prompt tokens and computes loss exclusively on assistant/tool targets.
4. Produces finite, non-NaN training loss values across all 10 steps.

---

## 4. Kaggle 50M-Token Execution

For single-session Kaggle T4x2 runs, `notebooks/kaggle-sft-v1-cell.py` executes:
1. Streaming and normalization of pinned public datasets.
2. Assembly of a deterministic 50,000-target mix.
3. 3,000 optimizer steps with FP16 precision.
4. Auto-export to `final_sft.zip` accompanied by complete provenance manifests and removal of temporary scratch caches.
