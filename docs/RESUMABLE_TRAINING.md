# Resumable pretraining and cooldown

Pretrain/cooldown checkpoints now contain model, optimizer, scheduler, GradScaler,
completed update count, consumed/updated token counts, best validation loss, and a
versioned traversal record. Every distributed rank contributes Python, NumPy,
PyTorch CPU and its local CUDA device's RNG. Seto uses one CUDA device per rank;
other devices are not initialized just to collect RNG. Saving is collective, with
rank 0 as the only writer. SFT/DPO continue using the compatible low-level APIs;
this change does **not** establish exact traversal replay for SFT/DPO.

## Resume semantics

`global_step` and the scheduler count **successful optimizer updates**.
`optimizer_attempts` also includes GradScaler skips; `skipped_updates` counts those
skips. `tokens_seen` counts all consumed input tokens, including skipped windows;
`tokens_updated` counts tokens used in successful updates. The cursor advances on
skips, so those batches are not replayed. Counts include all ranks. Loss logging
reports the latest accumulation window.

The sampler uses a private generator with the saved seed and epoch. The cursor is
the number of consumed full microbatches per rank, independent of worker
prefetching. Reconstructing an iterator slices sampler indices directly; it does
not fetch and discard already consumed examples. Only full equal-size batches per
rank are used. The shuffled tail that cannot form a full batch across all ranks
is omitted each epoch; samples are not padded with repeated indices. This is a
batching policy, not a resume-specific skip. A dataset too small for one full batch
per rank is rejected.

An accumulation window at epoch end may contain fewer microbatches than configured.
It is flushed using its actual count as the loss divisor. DDP `no_sync()` encloses
both forward and backward on earlier microbatches; the last microbatch synchronizes.
The saved cursor always points after a completed optimizer attempt, with zero
pending gradients. Epoch-boundary cursors normalize to the next epoch, batch zero.
Seto's fixed complex RoPE buffer is reconstructed from configuration; DDP buffer
broadcast is disabled because Gloo does not support broadcasting that dtype.

SIGTERM/SIGINT in the CLI request a stop at the next optimizer boundary, agreed
across ranks. A hard kill during an accumulation window loses work since the last
published checkpoint; restarting replays that unsaved work from the saved RNG and
cursor. It never claims to recover partially accumulated gradients. Fatal errors
propagate; they do not export a partial model as a successful training result.

## Compatibility and limits

Exact resume requires the same immutable ShardDataset contents (full SHA256 per
ordered shard, filename and sequence length), model configuration, batch and
accumulation settings, sampler seed, worker count, world size, optimizer/scaler
types, schedule horizon, epoch limit, precision and evaluation frequency. Runtime
checks include Python/PyTorch/NumPy, device/backend, CUDA/cuDNN, GPU name and relevant
determinism/TF32 flags. Full hashing costs an extra sequential dataset read when a
trainer is constructed; relocation to a different Kaggle mount path is allowed.

Output/data paths, rank IDs, logging/save frequency and retention may change.
Changing `max_steps` changes the cosine schedule and is rejected. Use
`--session-steps` to limit a session without changing the original schedule horizon.

The check happens before model/optimizer mutation. Legacy weight files, directory
checkpoints, and old ZIP checkpoints remain loadable through `load_checkpoint()` or
`--init-from`; exact pretrain/cooldown `--resume` explicitly rejects checkpoints
without the new traversal record. `--init-from` intentionally starts a new run/stage
with fresh optimizer, scheduler, scaler and cursor. It is the way to begin cooldown
from pretraining, **not** a way to resume an interrupted cooldown.

Supported replay tests use deterministic map-style ShardDataset with zero or two
workers. Other datasets without `resume_identity()` and iterable datasets fail
explicitly. A custom dataset implementing that opt-in protocol must provide a
stable content identity and pure, deterministic indexing; stochastic transforms,
worker-local random state, mutable/streaming sources and side effects are not
covered by exact replay. Worker count or world-size changes are rejected; there is
no automatic non-exact resume fallback.

CPU tests demonstrate bitwise replay on the tested runtime. CUDA/NCCL/FP16 on T4
was **not run**. CUDA RNG is saved per rank, but CUDA kernel nondeterminism, changed
hardware/drivers/libraries, or external nondeterministic code can prevent bitwise
numeric replay even with identical state/data. Do not interpret the checks as a
GPU reproducibility benchmark or a guarantee across environments. Only standard
GradScaler scale-backoff semantics are supported for skip detection. The CPU test
uses a simulated scaler to exercise a skipped update; it is not a real FP16 test.

## Crash safety and disk budget

A checkpoint is written directly into a unique hidden temporary file in the target
directory. All tensor ZIP entries permit ZIP64. The writer closes and fsyncs it,
checks required entries, metadata/manifest and every entry CRC, then uses
`os.replace()` to publish it. Directory fsync is attempted where supported. Local
POSIX atomic rename is the supported deployment target; remote filesystems must
provide equivalent semantics. Failed writes remove their temporary file in
`finally` and leave published archives unchanged. A hard kill can leave hidden
`.tmp` files, which auto-resume ignores. Auto-discovery checks ZIP integrity instead
of trusting a filename/mtime; if no valid checkpoint exists but checkpoint artifacts
are present, the CLI refuses to silently start a new traversal.

Retention runs **after** successful publication and only rotates archives created
by this Python process. `keep_last_n` must be at least one. Pre-existing user ZIPs
and legacy directories are never removed by automatic retention. `--clean` is
rejected for pretrain/cooldown. Independent writers must not share one checkpoint
directory; only DDP rank 0 writes.

For checkpoint size S, saving needs retained archives **plus another S** while the
temporary replacement is written, including replacement of the same step. With
one current-process checkpoint retained, budget roughly **2S** in that directory,
plus all pre-existing archives and any best-checkpoint directory. Best checkpoints
are full independent resumable archives, so budget their own retained copy and
replacement. There is no model-directory/ZIP duplication during checkpoint writes.
The free-space preflight conservatively estimates tensor storage plus overhead
(and may overcount tied tensors). ENOSPC or validation/rename failure aborts saving
and is reported to all ranks; it never frees space by deleting an old valid file
first. If needed, move verified old session copies yourself to persistent storage
before the next run. Abandoned temporary files may also need manual removal after
ensuring there is no active writer.

Loading uses unique automatically cleaned seekable temporary files. Two ranks can
load concurrently without sharing names or unlinking each other's files. Temporary
disk demand is up to **world_size × largest extracted entry** (usually optimizer
state), in addition to retained checkpoints. Set `TMPDIR` before Python starts to
select a sufficiently large local filesystem. Final model export is separate from
resume and can require an additional model directory plus model ZIP; use
`--skip-export` for session runs to avoid those copies. `final_pretrain.zip` and
`final_cooldown.zip` are weights-only exports, not resume checkpoints.

Only load trusted checkpoint files: optimizer/RNG state uses PyTorch pickle loading.
Integrity validation detects accidental corruption, not malicious content.

## Next Kaggle session (two T4 processes)

Keep the latest validated `seto_step_XXXXXXXX.zip` in persistent Kaggle output and
attach it read-only to the next session. Keep the same code/runtime, tokenizer,
shard contents/names and training settings. Do not use notebook helpers that delete
checkpoints or substitute `--init-from` for `--resume`.

Example initial run (choose the total schedule horizon once):

```bash
mkdir -p /kaggle/working/seto-tmp
export TMPDIR=/kaggle/working/seto-tmp
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --stage pretrain --model-config tiny --data-dir /kaggle/input/seto-shards \
  --tokenizer /kaggle/input/seto-tokenizer --output-dir /kaggle/working/run \
  --batch-size 2 --grad-accum 4 --seq-len 1024 --num-workers 2 --seed 42 \
  --max-steps 10000 --session-steps 500 --save-every 100 --skip-export
```

Next session: use the same command/settings, add
`--resume /kaggle/input/previous-session/run/checkpoints_pretrain/seto_step_00000500.zip`,
and write to a fresh `/kaggle/working/run` directory. `--session-steps 500` then
allows up to 500 **additional** successful updates, capped by the unchanged 10000
step horizon. You can also reuse a writable checkpoint directory for validated
auto-resume. Imported previous-session files are not removed by retention.

For an experimental model supply its original direct ModelConfig JSON through
`--model-config-file`. All saved model/training fields can be inspected without
loading weight tensors:

```python
import json, zipfile
with zipfile.ZipFile(checkpoint_path) as archive:
    metadata = json.loads(archive.read("meta.json"))
    print(metadata["step"], metadata["tokens_seen"], metadata["config"])
```

To start cooldown, use `--stage cooldown --init-from <pretrain-checkpoint>` and a
new output directory. To continue that cooldown in a later session, use
`--stage cooldown --resume <cooldown-checkpoint>` with the original cooldown settings.

## Validation

Run `python -m pytest tests/test_resumable_training.py -q`. Tests cover interrupted
versus uninterrupted weights/optimizer/scheduler/RNG/sample order, worker prefetch,
mid-epoch and epoch-boundary cursors, partial windows, injected write/validation/
rename/low-space failures, same-step replacement, incomplete archive discovery,
concurrent loads and exception cleanup, legacy loading, incompatible resume,
best-checkpoint state, retention safety and simulated GradScaler skips.

The two-process CPU Gloo test checks simultaneous rank loading, exact per-rank
resume, gradient synchronization count and agreement with a single-process
concatenated-batch reference, including incomplete accumulation windows.

Verification in this workspace: Python 3.12.15, PyTorch 2.2.2, NumPy 1.26.4.
The full collected suite passed 74 tests; two subsequently added cooldown/CRC
regressions passed separately, bringing coverage to 76 tests (27 resumable-training
cases). The final Gloo/reference test also passed after the runtime metadata update.
Focused Ruff checks, syntax compilation, CLI help and diff whitespace checks pass.
There were no GPU runs or external dataset downloads.
