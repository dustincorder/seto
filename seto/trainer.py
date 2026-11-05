"""Seto trainer — pretraining with FP16, SDPA, full checkpoints."""

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import load_checkpoint, read_checkpoint_metadata, save_checkpoint
from .config import TrainConfig
from .data import ResumableBatchSampler


def setup_distributed(local_rank: int):
    if local_rank == -1:
        return
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    # torchrun already sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    # Only call init_process_group if torchrun didn't already do it
    if not dist.is_initialized():
        # Rank 0 can spend several minutes writing multi-gigabyte optimizer state.
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                timeout=timedelta(hours=1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_schedule(optimizer, warmup_steps: int, max_steps: int, min_lr: float, base_lr: float):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr / base_lr + 0.5 * (1.0 - min_lr / base_lr) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class SetoTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        config: Optional[TrainConfig] = None,
        local_rank: int = -1,
    ):
        self.config = config or TrainConfig()
        self.local_rank = local_rank
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.config.world_size = self.world_size
        self.is_main = self.rank == 0

        if local_rank >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        model = model.to(self.device)

        if dist.is_initialized():
            model = DDP(model, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                        # Seto's only buffer is deterministic complex RoPE; Gloo
                        # cannot broadcast this dtype and it has no mutable state.
                        broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=25)
        self.model = model

        adam_kwargs = dict(
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        if self.device.type == 'cuda':
            optimizer_created = False
            try:
                import bitsandbytes as bnb
                if hasattr(bnb.optim, "PagedAdamW8bit"):
                    self.optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), **adam_kwargs)
                    if self.is_main:
                        print("[trainer.py] Optimizer: Paged 8-bit AdamW (bitsandbytes) — saving ~6 GB VRAM with CPU paging", flush=True)
                else:
                    self.optimizer = bnb.optim.AdamW8bit(model.parameters(), **adam_kwargs)
                    if self.is_main:
                        print("[trainer.py] Optimizer: 8-bit AdamW (bitsandbytes) — saving ~6 GB VRAM", flush=True)
                optimizer_created = True
            except Exception:
                pass

            if not optimizer_created:
                try:
                    self.optimizer = torch.optim.AdamW(model.parameters(), fused=True, **adam_kwargs)
                except Exception:
                    self.optimizer = torch.optim.AdamW(model.parameters(), **adam_kwargs)
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), **adam_kwargs)

        # FP16 for T4 (Turing) — bf16 not supported on T4
        use_amp = (self.config.use_fp16 or self.config.use_bf16) and self.device.type == "cuda"
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if self.config.use_bf16 else torch.float16
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and self.config.use_fp16)
        else:  # torch 2.1/2.2 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp and self.config.use_fp16)

        self.scheduler = get_cosine_schedule(
            self.optimizer, self.config.warmup_steps, self.config.max_steps,
            self.config.min_lr, self.config.lr
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.global_step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        self.epoch = 0
        self.batch_cursor = 0
        self.optimizer_attempts = 0
        self.skipped_updates = 0
        self.tokens_updated = 0
        self.stop_requested = False
        self._at_step_boundary = True

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            alloc_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
            res_gb = torch.cuda.memory_reserved(self.device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            free_gb = total_gb - alloc_gb
            print(f"[rank {self.local_rank}] Device: {self.device} | VRAM allocated: {alloc_gb:.2f} GB | Reserved: {res_gb:.2f} GB | Free: {free_gb:.2f} GB / {total_gb:.2f} GB", flush=True)

        self._setup_dataloader()

    def _setup_dataloader(self):
        from torch.utils.data import DataLoader, IterableDataset
        if isinstance(self.train_dataset, IterableDataset) or not callable(
            getattr(self.train_dataset, "resume_identity", None)
        ):
            raise ValueError("Exact pretrain/cooldown traversal requires deterministic map-style "
                             "data with resume_identity(); use ShardDataset")
        if self.config.batch_size < 1 or self.config.grad_accum_steps < 1 or self.config.num_workers < 0:
            raise ValueError("Invalid batch/accumulation/worker configuration")
        self.dataset_identity = self.train_dataset.resume_identity()
        self.val_identity = self.val_dataset.resume_identity() if self.val_dataset is not None else None
        self.train_sampler = ResumableBatchSampler(
            self.train_dataset, self.config.batch_size, self.config.seed, self.rank, self.world_size)
        # DataLoader iterator construction must not consume the model's CPU RNG.
        self.loader_generator = torch.Generator()
        self.train_loader = DataLoader(
            self.train_dataset, batch_sampler=self.train_sampler,
            num_workers=self.config.num_workers, pin_memory=self.device.type == "cuda",
            generator=self.loader_generator, persistent_workers=False,
        )
        self.val_loader = None
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=self.config.batch_size, shuffle=False,
                num_workers=self.config.num_workers, generator=torch.Generator().manual_seed(self.config.seed),
            )

    def _resume_contract(self):
        # Paths, rank IDs, reporting/save frequency may change between Kaggle sessions.
        fields = ("stage", "lr", "min_lr", "beta1", "beta2", "eps", "weight_decay",
                  "max_grad_norm", "warmup_steps", "max_steps", "epochs", "lr_schedule",
                  "batch_size", "grad_accum_steps", "use_fp16", "use_bf16", "num_workers",
                  "max_seq_len", "seed", "eval_every")
        raw = self.model.module if hasattr(self.model, "module") else self.model
        return json.loads(json.dumps(dict(
            training={name: getattr(self.config, name) for name in fields},
            model=asdict(raw.config), dataset=self.dataset_identity, validation=self.val_identity,
            optimizer=f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}",
            scaler=f"{type(self.scaler).__module__}.{type(self.scaler).__qualname__}",
            world_size=self.world_size, runtime=dict(
                torch=str(torch.__version__), numpy=np.__version__, python=sys.version,
                device=self.device.type,
                gpu=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                backend=dist.get_backend() if dist.is_initialized() else None,
                cuda=torch.version.cuda, cuda_devices=torch.cuda.device_count(),
                cudnn=torch.backends.cudnn.version(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
            ),
        )))

    def resume(self, checkpoint_path: str):
        """Exact resume only. Legacy/model-only archives must use init_from()."""
        if not str(checkpoint_path).endswith(".zip"):
            raise ValueError("Legacy checkpoints lack exact traversal state; use --init-from")
        meta = read_checkpoint_metadata(checkpoint_path)
        state = meta.get("training", {})
        if state.get("version") != 1:
            raise ValueError("Checkpoint lacks exact resume state; use --init-from for a new stage/run")
        if state.get("contract") != self._resume_contract():
            raise ValueError("Incompatible exact resume dataset/configuration/runtime; "
                             "keep the original settings or use --init-from (new run)")
        for name in ("scheduler", "scaler", "rng"):
            if name not in meta:
                raise ValueError(f"Exact resume checkpoint missing {name}")
        epoch, cursor = state["epoch"], state["batch_cursor"]
        total = self.train_sampler.total_batches
        if (type(epoch) is not int or epoch < 0 or type(cursor) is not int or not 0 <= cursor < total
                or cursor % self.config.grad_accum_steps):
            raise ValueError("Invalid optimizer-boundary data cursor")
        ranks = meta["rng"].get("ranks", [])
        if len(ranks) != self.world_size:
            raise ValueError("Missing rank-specific RNG states")
        load_checkpoint(checkpoint_path, self.model, self.optimizer, str(self.device))
        self.scheduler.load_state_dict(meta["scheduler"])
        self.scaler.load_state_dict(meta["scaler"])
        self.global_step = meta["step"]
        self.tokens_seen = meta["tokens_seen"]
        self.best_val_loss = state["best_val_loss"]
        self.epoch, self.batch_cursor = epoch, cursor
        self.optimizer_attempts = state["optimizer_attempts"]
        self.skipped_updates = state["skipped_updates"]
        self.tokens_updated = state["tokens_updated"]
        self._restore_rng(ranks[self.rank])
        self._at_step_boundary = True
        if self.is_main:
            print(f"Exact resume: step={self.global_step}, epoch={epoch}, next_batch={cursor}")

    def init_from(self, checkpoint_path: str):
        """Start a new run/stage from weights, without importing any progress or RNG."""
        if self.global_step or self.optimizer_attempts:
            raise ValueError("init_from requires a fresh trainer")
        load_checkpoint(checkpoint_path, self.model, device=str(self.device), allow_vocab_growth=True)
        if self.is_main:
            print(f"Initialized weights from {checkpoint_path}; optimizer and traversal start fresh")

    def _capture_rng(self):
        # One CUDA device per DDP process; do not initialize other ranks' GPUs.
        return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)

    def _restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["cpu"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)

    def _optimizer_step(self):
        """Return whether an update occurred; GradScaler may skip non-finite gradients."""
        if self.config.use_fp16:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        if self.config.use_fp16:
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            updated = self.scaler.get_scale() >= old_scale
        else:
            self.optimizer.step()
            updated = True
        self.optimizer.zero_grad(set_to_none=True)
        if dist.is_initialized():
            result = torch.tensor(int(updated), device=self.device)
            dist.all_reduce(result)
            if result.item() not in (0, self.world_size):
                raise RuntimeError("Ranks disagreed on optimizer update; refusing to checkpoint")
        return updated

    def train(self, session_steps: Optional[int] = None):
        """Train to the configured horizon, optionally stop this session at a boundary."""
        from contextlib import nullcontext
        if session_steps is not None and session_steps < 1:
            raise ValueError("session_steps must be positive")
        stop_step = min(self.config.max_steps, self.global_step + session_steps) if session_steps else self.config.max_steps
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        last_loss = 0.0
        stop_training = False
        while self.global_step < stop_step and not stop_training:
            if self.config.epochs is not None and self.epoch >= self.config.epochs:
                break
            self.train_sampler.epoch = self.epoch
            self.train_sampler.start_batch = self.batch_cursor
            self.loader_generator.manual_seed(self.config.seed + self.epoch)
            iterator = iter(self.train_loader)
            total = self.train_sampler.total_batches
            while self.batch_cursor < total and self.global_step < stop_step:
                # Flush partial accumulation windows at epoch end, scaling by the
                # actual number of full microbatches, never by a larger nominal size.
                window = min(self.config.grad_accum_steps, total - self.batch_cursor)
                self._at_step_boundary = False
                tokens = 0
                window_loss = 0.0
                for micro in range(window):
                    batch = next(iterator)
                    ids = batch["input_ids"].to(self.device, non_blocking=True)
                    labels = batch["labels"].to(self.device, non_blocking=True)
                    tokens += ids.numel()
                    sync = micro == window - 1
                    context = self.model.no_sync() if isinstance(self.model, DDP) and not sync else nullcontext()
                    with context:  # DDP forward AND backward must be inside no_sync.
                        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                            _, loss = self.model(ids, targets=labels)
                            loss = loss / window
                        if self.config.use_fp16:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    window_loss += loss.detach().item()
                updated = self._optimizer_step()
                self.batch_cursor += window
                self.optimizer_attempts += 1
                tokens *= self.world_size
                self.tokens_seen += tokens  # Includes consumed data on skipped updates.
                if updated:
                    self.scheduler.step()
                    self.global_step += 1
                    self.tokens_updated += tokens
                else:
                    self.skipped_updates += 1
                # Normalize the cursor before saving, including epoch-boundary saves.
                epoch_done = self.batch_cursor == total
                if epoch_done:
                    self.epoch += 1
                    self.batch_cursor = 0
                self._at_step_boundary = True
                last_loss = window_loss
                if updated and self.global_step % self.config.log_every == 0 and self.is_main:
                    print(f"Step {self.global_step} | loss={last_loss:.4f} | "
                          f"tokens={self.tokens_seen} | elapsed={time.time()-start_time:.1f}s")
                best = False
                if updated and self.global_step % self.config.eval_every == 0 and self.val_loader is not None:
                    value = self.evaluate()
                    if value < self.best_val_loss:
                        self.best_val_loss = value
                        best = True
                if best or (updated and self.global_step % self.config.save_every == 0):
                    self._save_checkpoint(last_loss, is_best=best)
                # Signals are reduced at a safe boundary; never serialize partial gradients.
                stop = torch.tensor(int(self.stop_requested), device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                stop_training = bool(stop.item())
                if epoch_done or stop_training:
                    break
        # Always persist the latest boundary, including consumed/skipped windows.
        self._save_checkpoint(last_loss)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(input_ids, targets=labels)
            total_loss += loss.item()
            n_batches += 1
        if dist.is_initialized():
            aggregate = torch.tensor([total_loss, n_batches], dtype=torch.float64, device=self.device)
            dist.all_reduce(aggregate)
            total_loss, n_batches = aggregate.tolist()
        self.model.train()
        return total_loss / max(1, n_batches)

    def _save_checkpoint(self, loss: float, is_best: bool = False):
        """Collective operation: every rank contributes its own RNG state."""
        if not self._at_step_boundary:
            raise RuntimeError("Cannot checkpoint a partial optimizer step")
        rng = self._capture_rng()
        ranks = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(ranks, rng)
        else:
            ranks[0] = rng
        raw = self.model.module if hasattr(self.model, "module") else self.model
        training = dict(version=1, contract=self._resume_contract(), epoch=self.epoch,
                        batch_cursor=self.batch_cursor, best_val_loss=self.best_val_loss,
                        optimizer_attempts=self.optimizer_attempts, skipped_updates=self.skipped_updates,
                        tokens_updated=self.tokens_updated)
        error = [None]
        if self.is_main:
            try:
                kwargs = dict(model=raw, optimizer=self.optimizer, scheduler=self.scheduler,
                              scaler=self.scaler, step=self.global_step, loss=loss,
                              config=dict(model=asdict(raw.config), training=asdict(self.config)),
                              rng_state=dict(ranks=ranks), tokens_seen=self.tokens_seen,
                              training_state=training)
                save_checkpoint(**kwargs, save_dir=self.config.checkpoint_dir,
                                keep_last_n=self.config.keep_last_n)
                if is_best:
                    save_checkpoint(**kwargs, save_dir=os.path.join(self.config.checkpoint_dir, "best"),
                                    keep_last_n=1)
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f"Checkpoint save failed; previous checkpoint retained: {error[0]}")
