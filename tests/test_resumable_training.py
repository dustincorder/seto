"""Tiny CPU resume tests; no external data or accelerators required."""

import copy
import json
import random
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from seto.checkpoint import (
    get_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from seto.config import ModelConfig, TrainConfig
from seto.data import ShardDataset
from seto.model import SetoLM
from seto.trainer import SetoTrainer


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def seed(value=123):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def shards(path):
    path.mkdir(parents=True, exist_ok=True)
    (np.arange(61, dtype=np.uint16) % 31).tofile(path / "train_000.bin")
    return path


class ObservedLM(SetoLM):
    def __init__(self, config):
        super().__init__(config)
        self.seen = []

    def forward(self, input_ids, targets=None, **kwargs):
        self.seen.extend(input_ids[:, 0].tolist())
        logits, loss = super().forward(input_ids, targets=targets, **kwargs)
        # Exercise Python/NumPy RNG as well as torch dropout in exact-replay tests.
        if loss is not None:
            loss = loss * (0.9 + random.random() * 0.05 + np.random.random() * 0.05)
        return logits, loss


def make_trainer(data_path, output, **changes):
    model = ObservedLM(
        ModelConfig(
            vocab_size=31,
            d_model=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=12,
            max_seq_len=4,
            dropout=0.2,
            use_gradient_checkpointing=False,
        )
    )
    config = TrainConfig(
        lr=0.01,
        min_lr=0.001,
        warmup_steps=0,
        max_steps=6,
        batch_size=2,
        grad_accum_steps=3,
        max_seq_len=4,
        use_fp16=False,
        num_workers=0,
        seed=17,
        weight_decay=0,
        log_every=100,
        eval_every=100,
        save_every=100,
        checkpoint_dir=str(output),
    )
    config = replace(config, **changes)
    return SetoTrainer(model, ShardDataset(str(data_path), seq_len=4), config=config)


def assert_state_equal(left, right, exact=True):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(
            left, right, rtol=0 if exact else 1e-5, atol=0 if exact else 1e-7
        )
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key], exact)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b, exact)
    else:
        assert left == right


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("interrupt_after", [1, 2])
def test_exact_training_resume(tmp_path, workers, interrupt_after):
    data = shards(tmp_path / "data")
    seed()
    full = make_trainer(data, tmp_path / "full", num_workers=workers)
    full.train()
    expected_rng = full._capture_rng()
    seed()
    part = make_trainer(data, tmp_path / "part", num_workers=workers)
    part.train(session_steps=interrupt_after)
    checkpoint = get_latest_checkpoint(str(tmp_path / "part"))
    # Disturb all RNG and reconstruct model/optimizer as in a fresh Kaggle session.
    seed(789)
    rest = make_trainer(data, tmp_path / "rest", num_workers=workers)
    rest.resume(checkpoint)
    rest.train()
    assert full.model.seen == part.model.seen + rest.model.seen
    assert_state_equal(full.model.state_dict(), rest.model.state_dict())
    assert_state_equal(full.optimizer.state_dict(), rest.optimizer.state_dict())
    assert_state_equal(full.scheduler.state_dict(), rest.scheduler.state_dict())
    assert_state_equal(expected_rng, rest._capture_rng())
    for key in (
        "global_step",
        "tokens_seen",
        "tokens_updated",
        "optimizer_attempts",
        "skipped_updates",
        "epoch",
        "batch_cursor",
        "best_val_loss",
    ):
        assert getattr(full, key) == getattr(rest, key)
    # Five full batches/epoch: accumulation windows are 3 then 2, never dropped.
    assert full.tokens_seen == 3 * 5 * 2 * 4


def checkpoint_args(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    return dict(
        model=model, optimizer=optimizer, step=1, loss=1.0, config={}, save_dir=str(tmp_path)
    )


@pytest.mark.parametrize("failure", ["write", "validate", "replace", "space"])
def test_failed_save_preserves_last_checkpoint(tmp_path, failure):
    args = checkpoint_args(tmp_path)
    old = Path(save_checkpoint(**args))
    original = old.read_bytes()
    args["step"] = 2
    target = {
        "write": "seto.checkpoint.torch.save",
        "validate": "seto.checkpoint.validate_checkpoint",
        "replace": "seto.checkpoint.os.replace",
        "space": "seto.checkpoint.shutil.disk_usage",
    }[failure]
    options = (
        dict(return_value=type("Disk", (), {"free": 0})())
        if failure == "space"
        else dict(side_effect=OSError("injected interruption"))
    )
    with patch(target, **options), pytest.raises(OSError):
        save_checkpoint(**args)
    assert old.read_bytes() == original
    assert get_latest_checkpoint(str(tmp_path)) == str(old)
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_same_step_interrupted_replacement(tmp_path):
    args = checkpoint_args(tmp_path)
    old = Path(save_checkpoint(**args))
    before = old.read_bytes()
    with patch("seto.checkpoint.os.replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError):
            save_checkpoint(**args)
    assert old.read_bytes() == before
    validate_checkpoint(old)


def test_auto_resume_ignores_incomplete_and_invalid(tmp_path):
    valid = save_checkpoint(**checkpoint_args(tmp_path))
    (tmp_path / "seto_step_99999999.zip").write_bytes(b"truncated")
    (tmp_path / "seto_step_99999998.tmp").write_bytes(b"partial")
    for step, meta in [(5, []), (6, {"step": -1, "config": {}})]:
        with zipfile.ZipFile(tmp_path / f"seto_step_{step:08d}.zip", "w") as zf:
            zf.writestr("meta.json", json.dumps(meta))
            zf.writestr("model.pt", b"x")
            zf.writestr("optimizer.pt", b"x")
    assert get_latest_checkpoint(str(tmp_path)) == valid
    with zipfile.ZipFile(valid) as src:
        with zipfile.ZipFile(tmp_path / "seto_step_00000007.zip", "w") as dst:
            for name in src.namelist():
                if name != "optimizer.pt":
                    dst.writestr(name, src.read(name))
    assert get_latest_checkpoint(str(tmp_path)) == valid


def test_concurrent_loads_and_cleanup(tmp_path):
    args = checkpoint_args(tmp_path)
    path = save_checkpoint(**args)

    def load(_):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        load_checkpoint(path, model, optimizer)
        assert_state_equal(args["model"].state_dict(), model.state_dict())

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))
    # Temporary files must close even when deserialization raises.
    files = []
    original = tempfile.TemporaryFile

    def track(*args, **kwargs):
        stream = original(*args, **kwargs)
        files.append(stream)
        return stream

    with (
        patch("seto.checkpoint.tempfile.TemporaryFile", side_effect=track),
        patch("seto.checkpoint._safe_torch_load", side_effect=RuntimeError("decode failed")),
        pytest.raises(RuntimeError),
    ):
        load(0)
    assert files and all(stream.closed for stream in files)


def test_legacy_load_and_no_deletion(tmp_path):
    args = checkpoint_args(tmp_path)
    folder = tmp_path / "step_00000000"
    folder.mkdir()
    torch.save(args["model"].state_dict(), folder / "model.pt")
    torch.save(args["optimizer"].state_dict(), folder / "optimizer.pt")
    (folder / "meta.json").write_text(json.dumps({"step": 0, "config": {}}))
    model = torch.nn.Linear(2, 2)
    load_checkpoint(str(folder), model, torch.optim.AdamW(model.parameters()))
    assert_state_equal(model.state_dict(), args["model"].state_dict())
    # Legacy nested ZIPs retain explicit loading support.
    legacy_zip = tmp_path / "legacy.zip"
    with zipfile.ZipFile(legacy_zip, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"legacy/{path.name}")
    load_checkpoint(str(legacy_zip), model)
    assert_state_equal(model.state_dict(), args["model"].state_dict())
    before = legacy_zip.read_bytes()
    save_checkpoint(**args)
    save_checkpoint(**(args | {"step": 2}))
    assert folder.exists() and legacy_zip.read_bytes() == before


@pytest.mark.parametrize("change", ["dataset", "workers", "world", "batch", "horizon", "seed"])
def test_incompatible_resume_rejected_before_model_mutation(tmp_path, change):
    data = shards(tmp_path / "data")
    seed()
    part = make_trainer(data, tmp_path / "part")
    part.train(session_steps=1)
    changes = {
        "workers": dict(num_workers=1),
        "batch": dict(batch_size=1),
        "horizon": dict(max_steps=7),
        "seed": dict(seed=18),
    }.get(change, {})
    if change == "dataset":
        # Same filename/size, different content must be detected.
        payload = np.fromfile(data / "train_000.bin", dtype=np.uint16)
        payload[0] = 30
        payload.tofile(data / "train_000.bin")
    rest = make_trainer(data, tmp_path / "rest", **changes)
    if change == "world":
        rest.world_size = 2
    before = copy.deepcopy(rest.model.state_dict())
    with pytest.raises(ValueError, match="Incompatible"):
        rest.resume(get_latest_checkpoint(str(tmp_path / "part")))
    assert_state_equal(before, rest.model.state_dict())


class SkipOnceScaler:
    def __init__(self):
        self.value = 16.0
        self.skip = True

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        if not self.skip:
            optimizer.step()

    def update(self):
        if self.skip:
            self.value /= 2
            self.skip = False

    def get_scale(self):
        return self.value

    def state_dict(self):
        return dict(value=self.value, skip=self.skip)

    def load_state_dict(self, state):
        self.__dict__.update(state)


def test_skipped_update_progress_and_resume(tmp_path):
    data = shards(tmp_path / "data")
    seed()
    trainer = make_trainer(data, tmp_path / "ckpt", use_fp16=True)
    trainer.scaler = SkipOnceScaler()
    initial_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
    initial_weights = copy.deepcopy(trainer.model.state_dict())
    # Stop after the first attempted (skipped) window via a forward hook.
    trainer.model.register_forward_hook(lambda *args: setattr(trainer, "stop_requested", True))
    trainer.train()
    assert trainer.global_step == 0 and trainer.optimizer_attempts == trainer.skipped_updates == 1
    assert trainer.tokens_seen == 24 and trainer.tokens_updated == 0
    assert trainer.batch_cursor == 3
    assert_state_equal(initial_weights, trainer.model.state_dict())
    assert_state_equal(initial_scheduler, trainer.scheduler.state_dict())
    rest = make_trainer(data, tmp_path / "rest", use_fp16=True)
    rest.scaler = SkipOnceScaler()
    rest.resume(get_latest_checkpoint(str(tmp_path / "ckpt")))
    assert rest.scaler.get_scale() == 8 and not rest.scaler.skip
    rest.train(session_steps=1)
    assert rest.global_step == 1 and rest.optimizer_attempts == 2 and rest.skipped_updates == 1
    assert rest.tokens_seen == 40 and rest.tokens_updated == 16
    assert rest.scheduler.last_epoch == 1


def test_legacy_requires_explicit_new_run(tmp_path):
    data = shards(tmp_path / "data")
    trainer = make_trainer(data, tmp_path / "ckpt")
    path = save_checkpoint(trainer.model, trainer.optimizer, 0, 1.0, {}, str(tmp_path / "legacy"))
    with pytest.raises(ValueError, match="lacks exact resume"):
        trainer.resume(path)
    trainer.init_from(path)
    assert trainer.global_step == trainer.batch_cursor == 0
    assert not trainer.optimizer.state


def test_partial_checkpoint_refused(tmp_path):
    trainer = make_trainer(shards(tmp_path / "data"), tmp_path / "ckpt")
    trainer._at_step_boundary = False
    with pytest.raises(RuntimeError, match="partial"):
        trainer._save_checkpoint(1.0)


def _ddp_trainer(data, output, steps=4):
    from seto.trainer import SetoTrainer

    model = SetoLM(
        ModelConfig(
            vocab_size=31,
            d_model=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=12,
            max_seq_len=4,
            dropout=0.0,
            use_gradient_checkpointing=False,
        )
    )
    config = TrainConfig(
        lr=0.01,
        min_lr=0.001,
        warmup_steps=0,
        max_steps=steps,
        batch_size=1,
        grad_accum_steps=3,
        max_seq_len=4,
        use_fp16=False,
        num_workers=0,
        seed=17,
        weight_decay=0,
        log_every=100,
        eval_every=100,
        save_every=100,
        checkpoint_dir=str(output),
    )
    return SetoTrainer(model, ShardDataset(str(data), seq_len=4), config=config)


def _gloo_worker(rank, root, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    root = Path(root)
    try:
        seed(123 + rank)
        full = _ddp_trainer(root / "data", root / "ddp-full")
        seen = []
        full.model.module.register_forward_pre_hook(
            lambda _, args: seen.extend(args[0][:, 0].tolist())
        )
        calls = []

        def hook(state, bucket):
            calls.append(1)
            return default_hooks.allreduce_hook(state, bucket)

        full.model.register_comm_hook(None, hook)
        full.train()
        # One gradient bucket per completed window (forward outside no_sync would
        # trigger an all-reduce for every microbatch and fail this check).
        assert len(calls) == 4
        full_rng = full._capture_rng()
        expected_model = copy.deepcopy(full.model.module.state_dict())
        expected_optimizer = copy.deepcopy(full.optimizer.state_dict())
        expected_scheduler = copy.deepcopy(full.scheduler.state_dict())
        seed(123 + rank)
        part = _ddp_trainer(root / "data", root / "ddp-part")
        part_seen = []
        part.model.module.register_forward_pre_hook(
            lambda _, args: part_seen.extend(args[0][:, 0].tolist())
        )
        part.train(session_steps=1)
        seed(999 + rank)
        rest = _ddp_trainer(root / "data", root / "ddp-rest")
        rest_seen = []
        rest.model.module.register_forward_pre_hook(
            lambda _, args: rest_seen.extend(args[0][:, 0].tolist())
        )
        rest.resume(str(root / "ddp-part" / "seto_step_00000001.zip"))
        # Both ranks concurrently load the exact same archive.
        rest.train()
        assert seen == part_seen + rest_seen
        assert_state_equal(expected_model, rest.model.module.state_dict())
        assert_state_equal(expected_optimizer, rest.optimizer.state_dict())
        assert_state_equal(expected_scheduler, rest.scheduler.state_dict())
        assert_state_equal(full_rng, rest._capture_rng())
        assert rest.epoch == full.epoch == 2
        assert rest.batch_cursor == full.batch_cursor == 0
        assert rest.tokens_seen == full.tokens_seen == 80
        torch.save(
            dict(
                model=expected_model,
                optimizer=expected_optimizer,
                scheduler=expected_scheduler,
                seen=seen,
            ),
            root / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="PyTorch was built without Gloo",
)
def test_two_process_gloo_resume_and_accumulation_reference(tmp_path, monkeypatch):
    import sys

    import torch.multiprocessing as mp

    from seto.data import ResumableBatchSampler

    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    shards(tmp_path / "data")
    mp.spawn(_gloo_worker, args=(str(tmp_path), str(tmp_path / "rendezvous")), nprocs=2, join=True)
    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False) for rank in range(2)]
    assert_state_equal(ranks[0]["model"], ranks[1]["model"])
    # Single-process reference applies each update to the concatenation of both
    # rank-local windows. Last window is 2 microbatches/rank, not the nominal 3.
    seed(123)
    reference = _ddp_trainer(tmp_path / "data", tmp_path / "reference")
    dataset = reference.train_dataset
    expected_seen = [[], []]
    for epoch in range(2):
        per_rank = []
        for rank in range(2):
            sampler = ResumableBatchSampler(dataset, 1, 17, rank, 2)
            sampler.epoch = epoch
            batches = list(sampler)
            per_rank.append(batches)
            expected_seen[rank].extend(dataset[b[0]]["input_ids"][0].item() for b in batches)
        for start in (0, 3):
            indices = [b[0] for batches in per_rank for b in batches[start : start + 3]]
            ids = torch.stack([dataset[i]["input_ids"] for i in indices])
            labels = torch.stack([dataset[i]["labels"] for i in indices])
            reference.optimizer.zero_grad(set_to_none=True)
            _, loss = reference.model(ids, targets=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                reference.model.parameters(), reference.config.max_grad_norm
            )
            reference.optimizer.step()
            reference.scheduler.step()
    assert ranks[0]["seen"] == expected_seen[0] and ranks[1]["seen"] == expected_seen[1]
    assert_state_equal(reference.model.state_dict(), ranks[0]["model"], exact=False)
    assert_state_equal(reference.optimizer.state_dict(), ranks[0]["optimizer"], exact=False)
    assert_state_equal(reference.scheduler.state_dict(), ranks[0]["scheduler"])


def test_best_checkpoint_has_full_resume_state(tmp_path):
    data = shards(tmp_path / "data")
    trainer = make_trainer(data, tmp_path / "ckpt")
    trainer.train(session_steps=1)
    trainer.best_val_loss = 0.125
    trainer._save_checkpoint(1.0, is_best=True)
    rest = make_trainer(data, tmp_path / "rest")
    rest.resume(get_latest_checkpoint(str(tmp_path / "ckpt" / "best")))
    assert rest.best_val_loss == 0.125
    assert_state_equal(trainer.scaler.state_dict(), rest.scaler.state_dict())
    assert_state_equal(trainer.optimizer.state_dict(), rest.optimizer.state_dict())


def test_preexisting_archives_survive_retention(tmp_path):
    args = checkpoint_args(tmp_path)
    old = Path(save_checkpoint(**args))
    before = old.read_bytes()
    # Simulate a separate session: prior files are not owned by this writer.
    with patch("seto.checkpoint._MANAGED_CHECKPOINTS", set()):
        second = Path(save_checkpoint(**(args | {"step": 2})))
        third = Path(save_checkpoint(**(args | {"step": 3})))
        assert not second.exists() and third.exists()
    assert old.read_bytes() == before


def test_unsupported_dataset_fails_explicitly(tmp_path):
    trainer = make_trainer(shards(tmp_path / "data"), tmp_path / "ckpt")
    dataset = torch.utils.data.TensorDataset(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="deterministic map-style"):
        SetoTrainer(SetoLM(trainer.model.config), dataset, config=trainer.config)


def test_cooldown_resume(tmp_path):
    data = shards(tmp_path / "data")
    seed()
    full = make_trainer(data, tmp_path / "full", stage="cooldown", max_steps=3)
    full.train()
    seed()
    part = make_trainer(data, tmp_path / "part", stage="cooldown", max_steps=3)
    part.train(session_steps=1)
    rest = make_trainer(data, tmp_path / "rest", stage="cooldown", max_steps=3)
    rest.resume(get_latest_checkpoint(str(tmp_path / "part")))
    rest.train()
    assert_state_equal(full.model.state_dict(), rest.model.state_dict())
    assert_state_equal(full.optimizer.state_dict(), rest.optimizer.state_dict())
    assert_state_equal(full.scheduler.state_dict(), rest.scheduler.state_dict())
    assert full.model.seen == part.model.seen + rest.model.seen


def test_crc_corruption_falls_back_to_previous_valid_checkpoint(tmp_path):
    import struct

    args = checkpoint_args(tmp_path) | {"keep_last_n": 2}
    old = save_checkpoint(**args)
    newer = Path(save_checkpoint(**(args | {"step": 2})))
    with zipfile.ZipFile(newer) as archive:
        offset = archive.getinfo("model.pt").header_offset
    content = bytearray(newer.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", content, offset + 26)
    content[offset + 30 + name_length + extra_length + 32] ^= 1
    newer.write_bytes(content)
    with pytest.raises(ValueError, match="CRC validation failed"):
        validate_checkpoint(newer)
    assert get_latest_checkpoint(str(tmp_path)) == old
