# Seto Restart Kit Review and Integration Assessment

> **Status:** Assessment and Recommendations  
> **Date:** November 2026  
> **Analyzed Version:** `dustincorder/seto` (`master` / `chore/repository-cleanup`)  
> **Source Examined:** `other/` (Seto restart kit overlay and documentation)

---

## 1. Executive Summary & Inventory

The `other/` directory contains an externally prepared "restart kit" generated on November 4, 2026. The kit provides concrete static audits of the original training codebase, data source catalogues, experimental configuration profiles, budget calculation tooling, and OpenAI Codex multi-agent role definitions.

### Complete Inventory of `other/`
- **`README.md` & `VALIDATION.md`**: Kit installation instructions, validation assertions, AST checks, parameter arithmetic reproduction.
- **`install.py`**: An automated installer script designed to overlay materials into the repository root. *(Marked SKIP: do not run)*.
- **`overlay/docs/restart/`**:
  - `AUDIT.md`: In-depth static audit of `seto/` and `scripts/` (identified HF token logging in notebooks, incomplete resume states, DDP temporary file races, missing validation loops, and GradScaler overflow tracking).
  - `PLAN.md`: Phased execution roadmap (M0 to M5), hardware realities (Kaggle 2xT4 quotas), and storage layouts.
  - `DATASETS.md`: Data source analysis covering FineWeb-Edu, FineWeb2 (RU/UK), The Stack Smol XL, OASST2, Aya, and tool datasets.
  - `START_PROMPT.md` & `STATUS.md`: Initial task framing and status logging conventions.
- **`overlay/configs/restart/`**:
  - `seed-50m.json` & `base-100m.json`: 50M and 100M parameter configurations utilizing 32k vocabularies.
  - `dataset_registry.json`: Structured JSON catalog of pretraining and SFT data sources with license notes and split rules.
- **`overlay/scripts/restart_budget.py`**: Dependency-free parameter count and training duration calculator based on throughput scenarios.
- **`overlay/AGENTS.md` & `.codex/`**: Codex agent orchestration guidelines, concurrency rules, and four role configurations (`seto_training`, `seto_data`, `seto_runtime`, `seto_reviewer`).

---

## 2. Keep / Adapt / Skip Decisions

| Item | Decision | Rationale & Action Plan |
| :--- | :---: | :--- |
| **`docs/restart/AUDIT.md` (Code Audit findings)** | **KEEP** | **Critical engineering value.** Identifies real bugs in checkpointing, DDP file collisions, missing pretraining validation, and secret logging. Should be referenced to drive upcoming trainer stabilization tasks. |
| **`scripts/restart_budget.py`** | **KEEP** | Clean, standalone calculation script with no external dependencies (zero torch dependency). Accurately calculates parameter counts and wall-clock times under Kaggle/Colab constraints. Can be integrated into `scripts/`. |
| **`configs/restart/dataset_registry.json`** | **KEEP** | High-quality schema capturing source licenses (ODC-By, Apache-2.0, BigCode restrictions), dataset splits, and language shares. |
| **`docs/restart/DATASETS.md`** | **ADAPT** | Highly relevant dataset analysis (FineWeb2 RU/UK, FineWeb-Edu EN, Python code, OASST2, tool schemas). **Adaptation needed:** Align with Seto's updated focus on up to 2B parameters, tool calling/agentic workflows, and balanced EN/RU/UK sampling. |
| **`docs/restart/PLAN.md`** | **ADAPT** | Excellent methodology for zero-budget training on Kaggle/Colab. **Adaptation needed:** The kit focuses on 50M/100M pilot architectures; adapt to reflect Seto's ultimate target of Seto Tiny/Daily (up to 2B) and modern architectural patterns (Gemma-style innovations, RMSNorm, GQA) rather than treating 50M as the final model. |
| **`configs/restart/seed-50m.json` & `base-100m.json`** | **ADAPT** | Useful as fast smoke-test and pipeline-validation presets on free GPUs. However, they use an old 32k vocabulary assumption (current Seto uses 48k BPE), and reflect legacy Llama-like layer proportions. Adapt to test modern Gemma-like or updated GQA ratios. |
| **`overlay/AGENTS.md` & `.codex/agents/*.toml`** | **ADAPT** | The role separations (`seto_training`, `seto_data`, `seto_runtime`, `seto_reviewer`) are conceptually sound. **Adaptation needed:** Antigravity and OpenCode development environments use different agent definitions than proprietary Codex TOML configs. Adapt the instructions into general agent prompts or repository instructions. |
| **`install.py` & automated overlay installer** | **SKIP** | Blindly running the overlay installer risks polluting repository structure, creating duplicated directories, and introducing tool-specific Codex files. |
| **Legacy Llama configurations as final architecture** | **SKIP** | The kit's presets use classical RoPE/SwiGLU ratios. Seto's architecture is explicitly not finalized and seeks modern ideas from Gemma (e.g. GeGLU/SwiGLU variants, post/pre-norm scaling, larger head dimensions, sliding-window attention) and other efficient models. |

---

## 3. Potential Conflicts with Current Repository

1. **Vocabulary Size Discrepancy (32,768 vs. 48,000)**:
   - *Current Seto*: Has an existing 48,000 BPE vocabulary configured across `seto/config.py`, `scripts/prepare_data.py`, and `README.md` to support Russian, English, Ukrainian, and special tokens (`<|tool_call|>`, `<|tool_result|>`).
   - *Restart Kit*: Proposes 32,768 (`seed-50m` and `base-100m`). Adopting 32k would break tokenization of existing shards or truncate multilingual compression efficiency unless thoroughly benchmarked.
2. **Model Horizon & Target Scale (50M–100M vs. Seto Tiny up to 2B)**:
   - The restart kit treats 50M–100M as the primary target. Seto's explicit project direction is Seto Tiny / Daily (up to 2 billion parameters), using 50M–100M purely as quick smoke-test / pipeline verification fixtures.
3. **Agent Ecosystem Compatibility**:
   - The restart kit includes OpenAI Codex-specific project configs (`.codex/config.toml`, `.codex/agents/*.toml`). In an Antigravity / OpenCode workflow, these files are extraneous unless adapted to general contributor guidelines.
4. **Architectural Stagnation**:
   - The restart kit re-uses standard Llama-style blocks without exploring modern advancements from Gemma (such as head-dimension choices, dual-residual norms, or alternative activations).

---

## 4. Recommended Next Development Tasks (Prioritized)

### Phase 1: Core Training & Checkpoint Stabilization (P0 — Immediate)
- [ ] **Task 1.1: Fix HF Token / Secret Logging in Notebooks**
  - Ensure API keys/tokens are passed exclusively via environment variables and masked from logged command lines in `notebooks/kaggle-1b-run1-cell.py`.
- [ ] **Task 1.2: Eliminate DDP File Collisions in Checkpointing**
  - Refactor `seto/checkpoint.py` so DDP ranks do not write to shared un-namespaced temporary paths (`/tmp/seto_ckpt_model_load.pt`).
- [ ] **Task 1.3: Robust Resume with Data Cursor & RNG States**
  - Ensure `_save_checkpoint()` and `SetoTrainer` save/restore exact dataset cursor positions, epoch counts, Python/NumPy/Torch RNG states across all ranks, and GradScaler loss scales.
- [ ] **Task 1.4: Fix DDP `no_sync()` Scope**
  - Wrap both microbatch forward and backward passes inside `model.no_sync()` during gradient accumulation steps.

### Phase 2: Architecture Exploration & Benchmark (P1 — Core ML)
- [ ] **Task 2.1: Modern Architecture Evaluation (Gemma & Efficient LLM Insights)**
  - Investigate and prototype architectural improvements inspired by Gemma:
    - Normalization: RMSNorm with learnable unit offset ($1 + w$).
    - Head dimensions: Larger key/value head dimensions (e.g. 128 or 256) for attention stability.
    - Activation: GeGLU vs. SwiGLU trade-offs on edge devices.
    - Post-attention and post-FFN normalization scales.
- [ ] **Task 2.2: Standalone Parameter & Throughput Calculator**
  - Integrate `scripts/restart_budget.py` into Seto to evaluate memory/VRAM headroom and realistic wall-clock estimates for dual T4 setups.
- [ ] **Task 2.3: Validation Split Protocol in Pretraining**
  - Add explicit held-out multilingual validation tracking to `scripts/train.py` for pretraining/cooldown.

### Phase 3: Multilingual Data Pipeline & Tool Calling (P2)
- [ ] **Task 3.1: Tokenizer Vocabulary Benchmark (32k vs. 48k)**
  - Measure compression ratios (bytes-per-token) on English, Russian, and Ukrainian samples to finalize vocabulary size.
- [ ] **Task 3.2: Curate Multilingual Pretrain Shards with Verification Manifests**
  - Implement manifest hashing and disjoint train/val splits for FineWeb2 (RU/UK) and FineWeb-Edu (EN).
- [ ] **Task 3.3: Tool Calling & Function Execution Fixtures**
  - Expand `seto-tools-v0` synthetic fixtures (local device control, calculator, unit testing) with target loss masking for agentic behavior.

---

## 5. Model Delegation Matrix

| Task Category | Suitable for Faster / Lighter Models (Gemini Flash / OpenCode) | Requires Stronger / Frontier Reasoning Models (Gemini Pro / Claude Opus) |
| :--- | :--- | :--- |
| **Bug Fixes & Cleanups** | • Secret masking in notebooks.<br/>• Temporary file collision fix in `seto/checkpoint.py`.<br/>• Adding CLI arguments & `.gitignore` maintenance. | • Rewriting DDP gradient synchronization semantics.<br/>• Exact rank-specific RNG and data loader cursor resumption. |
| **Tooling & Scripts** | • Porting `scripts/restart_budget.py`.<br/>• Formatting dataset manifests and JSON registry schemas. | • Designing automated checkpoint recovery & timeout handling logic for Kaggle sessions. |
| **ML Architecture** | • Adding configuration dataclass fields.<br/>• Writing unit tests for layer parameter calculations. | • Evaluating and designing Gemma-style architectural modifications (attention scaling, head dimensions, numerical stability in FP16). |
| **Data & Training** | • Converting dataset schemas (JSONL message parsing).<br/>• Adding preflight dataset validator scripts. | • Multi-lingual tokenization trade-off analysis and loss-weight balancing across languages. |
