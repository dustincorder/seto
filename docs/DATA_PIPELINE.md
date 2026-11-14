# Seto Multilingual Dataset Pipeline & Audit

> **Status:** Implemented & Documented  
> **Date:** November 2026  
> **Languages:** Russian (RU), English (EN), Ukrainian (UK), Code (Python)

---

## 1. Overview & Architecture

The Seto pretraining data pipeline produces clean, uint16 memory-mapped binary shards for training tiny language models from scratch on zero-dollar GPU allocations (e.g. Kaggle 2×T4).

### Key Features
1. **Deterministic Document-Level Train/Validation Splitting**:
   - Uses SHA-256 hash modulo arithmetic on document content: `deterministic_document_split(doc_hash, val_ratio, salt)`.
   - Independent of dataset streaming order, chunk batch size, or worker count.
   - **Zero Cross-Split Duplication Guarantee**: A document's hash maps strictly to either the `train` or `val` split.
2. **Exact-Text Deduplication**:
   - In-memory SHA-256 deduplicator tracks all seen documents across all streaming sources.
   - **Documented Limitation**: This implements *exact Unicode string matching* only. It does not perform MinHash LSH, semantic embedding distance, or fuzzy editing distance deduplication. Minor whitespace variations, typos, or localized edits are treated as distinct documents.
3. **Separate Shard Production (`train_*.bin` & `val_*.bin`)**:
   - Tokens are packed into standalone, contiguous binary files containing little-endian `uint16` IDs (0 to 65,535).
   - Generates separate `train_0000.bin`, `train_0001.bin`, and `val_0000.bin` shards directly compatible with `ShardDataset`.
4. **Strict Split Fallback Protection**:
   - `ShardDataset` and `find_shards` enforce `strict_split=True` for validation.
   - If validation shards are missing, the loader **raises `FileNotFoundError`** immediately. It never silently falls back to training shards.
5. **Pretraining Validation Integration**:
   - `scripts/train.py` accepts `--val-data-dir` (or auto-detects `val_*.bin` in `--data-dir`).
   - Seamlessly connects `val_dataset` to `SetoTrainer` with exact resume contract compatibility (`resume_identity()`).
6. **Detailed Provenance Manifest (`manifest.json`)**:
   - Records source breakdown, raw documents read, filtered count, duplicate count, tokens per split, and per-shard SHA-256 hashes and byte lengths.
7. **Tokenizer Evaluation Utilities**:
   - Added `evaluate_tokenizer_efficiency()` in `seto/pipeline.py` to compare compression ratios (bytes per token, characters per token) across EN, RU, and UK without retraining the tokenizer.
   - Preserves default 48k BPE vocabulary compatibility for current checkpoints.

---

## 2. Dataset Audit: Sources, Licenses, and Access

The following audit verifies the dataset sources targeted by `scripts/prepare_data.py`. Verified facts are explicitly separated from unverified assumptions and legal caveats.

### Verified Facts & Primary Sources

| Dataset ID | Configuration / Split | License | Access Requirements | Verified Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **`HuggingFaceFW/fineweb-2`** | `rus_Cyrl` (train) | **ODC-By v1.0** (subject to Common Crawl Terms of Use) | Public / open access | Primary Russian pretraining corpus (~50% target share). |
| **`HuggingFaceFW/fineweb-2`** | `ukr_Cyrl` (train) | **ODC-By v1.0** (subject to Common Crawl Terms of Use) | Public / open access | Ukrainian pretraining corpus (~10–15% target share). |
| **`HuggingFaceFW/fineweb-edu`** | `sample-10BT` (train) | **ODC-By v1.0** (subject to Common Crawl Terms of Use) | Public / open access | Educational-quality English pretraining corpus (~30–40% target share). |
| **`wikimedia/wikipedia`** | `20231101.ru` (train) | **CC BY-SA 3.0 / GFDL** | Public / open access | High-quality encyclopedic Russian factual reference text. |
| **`bigcode/the-stack-smol-xl`** | `data_dir="data/python"` | **Per-repository open licenses** | Public / open access | Light Python code / everyday logic. |

### Unverified Assumptions & Caveats

1. **Wikipedia ShareAlike (CC BY-SA 3.0 / GFDL) Compliance**:
   - *Status*: **Legal uncertainty in commercial ML**. The Wikimedia Foundation licenses text under CC BY-SA 3.0. Whether training a language model weights creates a "derivative work" that triggers ShareAlike viral clauses remains an unsettled legal question across jurisdictions.
   - *Mitigation*: Seto isolates Wikipedia sampling to an optional flag (`--max-samples-wiki`), which can be set to `0` for purely permissive/ODC-By builds.
2. **The Stack Smol XL Code Licensing**:
   - *Status*: **Non-monolithic license**. The Stack aggregates code files from GitHub across various licenses (MIT, Apache, BSD, etc.). Some files may have attribution notices or opt-out requests.
   - *Mitigation*: BigCode provides an opt-out mechanism. The pipeline records file provenance hashes. If strict commercial cleanliness is required, use verified in-house permissively licensed repositories.
3. **Cyrillic Script Separation in FineWeb2**:
   - *Status*: FineWeb2 uses FastText language identification. Short texts occasionally misclassify Ukrainian as Russian or vice-versa.
   - *Mitigation*: Minimum character threshold filter (`min_chars=100`) discards ambiguous short snippets.

---

## 3. Pipeline Usage Guide

### Preparing Shards & Manifest
```bash
python scripts/prepare_data.py \
  --output-dir data \
  --tokenizer-dir seto-tokenizer \
  --val-ratio 0.05 \
  --shard-size 50000000 \
  --max-samples-ru 500000 \
  --max-samples-en 450000 \
  --max-samples-uk 80000 \
  --max-samples-technical 30000 \
  --max-samples-wiki 30000 \
  --skip-tokenizer
```

Outputs in `data/shards/`:
- `train_0000.bin`, `train_0001.bin`, ...
- `val_0000.bin`, `val_0001.bin`, ...
- `data/manifest.json`

### Training with Validation
```bash
python scripts/train.py \
  --stage pretrain \
  --model-config tiny \
  --data-dir data/shards \
  --output-dir output
```
*(Automatically detects `val_*.bin` in `data/shards` and activates periodic validation loss tracking in `SetoTrainer`).*
