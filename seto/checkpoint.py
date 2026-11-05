"""Crash-safe ZIP checkpoints; old checkpoints survive failed replacement writes."""

import json
import os
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Only rotate files created by this process; never delete pre-existing user artifacts.
_MANAGED_CHECKPOINTS = set()

VOCAB_WEIGHT_KEYS = {"tok_embeddings.weight", "output.weight"}


def _load_model_state(
    model: nn.Module,
    state_dict: dict,
    allow_vocab_growth: bool,
) -> None:
    if allow_vocab_growth:
        current = model.state_dict()
        state_dict = dict(state_dict)
        for key in VOCAB_WEIGHT_KEYS:
            if key not in state_dict or key not in current:
                continue
            saved = state_dict[key]
            target = current[key]
            if saved.shape == target.shape:
                continue
            can_grow = (
                saved.ndim == 2
                and target.ndim == 2
                and saved.shape[1] == target.shape[1]
                and saved.shape[0] < target.shape[0]
            )
            if not can_grow:
                raise RuntimeError(
                    f"Cannot adapt {key}: checkpoint {tuple(saved.shape)} -> "
                    f"model {tuple(target.shape)}"
                )
            expanded = target.clone()
            expanded[:saved.shape[0]].copy_(saved.to(expanded.device, expanded.dtype))
            state_dict[key] = expanded
    model.load_state_dict(state_dict)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: dict,
    save_dir: str,
    keep_last_n: int = 1,
    scheduler=None,
    scaler=None,
    rng_state: Optional[dict] = None,
    tokens_seen: int = 0,
    training_state: Optional[dict] = None,
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if keep_last_n < 1:
        raise ValueError("keep_last_n must be at least one")
    zip_path = save_dir / f"seto_step_{step:08d}.zip"
    raw_model = model.module if hasattr(model, "module") else model
    entries = {"model.pt": raw_model.state_dict(), "optimizer.pt": optimizer.state_dict()}
    for name, obj in (("scheduler.pt", scheduler), ("scaler.pt", scaler)):
        if obj is not None:
            entries[name] = obj.state_dict()
    if rng_state is not None:
        entries["rng.pt"] = rng_state
    if training_state is not None:
        entries["training.pt"] = training_state
    meta = dict(format_version=2, step=step, loss=loss, tokens_seen=tokens_seen,
                config=config, exact_resume=training_state is not None, required_entries=sorted([*entries, "meta.json"]))

    # A conservative estimate catches ordinary low-disk cases; ENOSPC is still handled
    # by the same rollback path if filesystem overhead/concurrent writers exceed it.
    required = 16 * 1024**2
    for value in entries.values():
        required += _tensor_bytes(value) + 4096 * _tensor_count(value)
    if shutil.disk_usage(save_dir).free < required:
        raise OSError(28, f"Checkpoint needs approximately {required} free bytes; old checkpoints retained")

    fd, temporary = tempfile.mkstemp(prefix=f".{zip_path.stem}-", suffix=".tmp", dir=save_dir)
    tmp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                for name, value in entries.items():
                    with zf.open(name, "w", force_zip64=True) as entry:
                        torch.save(value, entry)
                zf.writestr("meta.json", json.dumps(meta, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        validate_checkpoint(tmp_path)
        os.replace(tmp_path, zip_path)
        # Persist the directory entry on platforms supporting directory fsync.
        try:
            directory_fd = os.open(save_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        tmp_path.unlink(missing_ok=True)
    _MANAGED_CHECKPOINTS.add(zip_path.resolve())
    _cleanup_old_checkpoints(save_dir, keep_last_n, exclude=zip_path)
    return str(zip_path)


def _tensor_count(value):
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, dict):
        return sum(_tensor_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_count(v) for v in value)
    return 0


def _tensor_bytes(value):
    # Overcount aliases rather than underestimate the required space.
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _find_zip_entry(zf: zipfile.ZipFile, name: str) -> Optional[str]:
    names = zf.namelist()
    if name in names:
        return name
    matches = [
        n for n in names
        if (n == name or n.endswith("/" + name))
        and ("/tokenizer/" not in n if name != "tokenizer.json" else True)
    ]
    return matches[0] if matches else None


def _safe_torch_load(f, map_location="cpu"):
    import torch
    try:
        import importlib
        import sys
        _u = importlib.import_module("torch._utils")
        torch._utils = _u
        sys.modules["torch"]._utils = _u

        try:
            _c = importlib.import_module("torch._C")
            for attr in ["dtype", "device", "layout", "finfo"]:
                if hasattr(_c, attr) and not hasattr(torch, attr):
                    setattr(torch, attr, getattr(_c, attr))
                    setattr(sys.modules["torch"], attr, getattr(_c, attr))
        except Exception:
            pass

        def _safe_element_size(dtype):
            s = str(dtype).lower()
            if "int64" in s or "long" in s or "float64" in s or "double" in s:
                return 8
            elif "float32" in s or "int32" in s or (("float" in s or "int" in s) and "16" not in s and "8" not in s and "64" not in s):
                return 4
            elif "float16" in s or "half" in s or "bfloat16" in s or "int16" in s or "short" in s:
                return 2
            elif "int8" in s or "uint8" in s or "bool" in s or "byte" in s:
                return 1
            if hasattr(dtype, "itemsize"):
                return dtype.itemsize
            return 4

        _u._element_size = _safe_element_size
    except Exception:
        pass

    try:
        return torch.load(f, map_location=map_location, weights_only=False)
    except Exception:
        return torch.load(f, map_location=map_location)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    allow_vocab_growth: bool = False,
) -> dict:
    path = Path(checkpoint_path)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            # Explicit weight-only legacy exports may lack metadata; exact resume
            # separately requires validated training metadata before loading weights.
            model_entry = _find_zip_entry(zf, "model.pt")
            if not model_entry:
                raise FileNotFoundError(f"model.pt not found in zip: {path}")
            raw_model = model.module if hasattr(model, "module") else model
            _load_model_state(raw_model, _load_zip_tensor(zf, model_entry, device), allow_vocab_growth)
            opt_entry = _find_zip_entry(zf, "optimizer.pt")
            if optimizer is not None:
                if not opt_entry:
                    raise ValueError("Checkpoint has no optimizer state; use weights-only initialization")
                optimizer.load_state_dict(_load_zip_tensor(zf, opt_entry, device))
            return _zip_metadata(zf)

    # Legacy: plain directory or direct model file path
    model_file = path if path.is_file() else (path / "model.pt")
    state_dict = _safe_torch_load(model_file, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    _load_model_state(raw_model, state_dict, allow_vocab_growth)

    ckpt_dir = path if path.is_dir() else path.parent
    if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            _safe_torch_load(ckpt_dir / "optimizer.pt", map_location=device)
        )

    meta = {}
    if (ckpt_dir / "meta.json").exists():
        with open(ckpt_dir / "meta.json") as f:
            meta = json.load(f)

    if (ckpt_dir / "scheduler.pt").exists():
        meta["scheduler"] = _safe_torch_load(ckpt_dir / "scheduler.pt", map_location=device)

    if (ckpt_dir / "scaler.pt").exists():
        meta["scaler"] = _safe_torch_load(ckpt_dir / "scaler.pt", map_location=device)

    if (ckpt_dir / "rng.pt").exists():
        meta["rng"] = _safe_torch_load(ckpt_dir / "rng.pt", map_location=device)

    return meta


def _load_zip_tensor(zf, entry, device="cpu"):
    # Unique, automatically closed/unlinked even when extraction/deserialization fails.
    # A seekable file also avoids Python stream-unpickler compatibility issues.
    with tempfile.TemporaryFile(prefix="seto-load-") as temporary:
        with zf.open(entry) as source:
            shutil.copyfileobj(source, temporary)
        temporary.seek(0)
        return _safe_torch_load(temporary, map_location=device)


def _zip_metadata(zf):
    entry = _find_zip_entry(zf, "meta.json")
    meta = json.loads(zf.read(entry)) if entry else {}
    for name in ("scheduler", "scaler", "rng", "training"):
        entry = _find_zip_entry(zf, name + ".pt")
        if entry:
            # RNG byte tensors must remain on CPU, even for CUDA model loading.
            meta[name] = _load_zip_tensor(zf, entry)
    return meta


def validate_checkpoint(path):
    """Validate training archive structure, metadata and all entry CRCs (no unpickle)."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate checkpoint ZIP entries")
        for name in ("model.pt", "optimizer.pt", "meta.json"):
            if not _find_zip_entry(zf, name):
                raise ValueError(f"Checkpoint missing {name}")
        meta = json.loads(zf.read(_find_zip_entry(zf, "meta.json")))
        if not isinstance(meta, dict):
            raise ValueError("Invalid checkpoint metadata object")
        if (type(meta.get("step")) is not int or meta["step"] < 0
                or not isinstance(meta.get("config"), dict)):
            raise ValueError("Invalid checkpoint metadata")
        if meta.get("format_version") not in (None, 2):
            raise ValueError("Unsupported checkpoint format version")
        if meta.get("format_version") == 2:
            if type(meta.get("tokens_seen")) is not int or meta["tokens_seen"] < 0:
                raise ValueError("Invalid checkpoint token count")
            required = meta.get("required_entries")
            if not isinstance(required, list) or not all(isinstance(n, str) for n in required) or not {"model.pt", "optimizer.pt", "meta.json"} <= set(required):
                raise ValueError("Invalid checkpoint entry manifest")
            if meta.get("exact_resume") and not {"training.pt", "rng.pt", "scheduler.pt", "scaler.pt"} <= set(required):
                raise ValueError("Exact checkpoint lacks required resume entries")
            if any(name not in names for name in required):
                raise ValueError("Checkpoint missing a declared entry")
            if any(zf.getinfo(name).file_size == 0 for name in required):
                raise ValueError("Checkpoint contains an empty required entry")
        if zf.testzip() is not None:
            raise ValueError("Checkpoint CRC validation failed")
        return meta


def read_checkpoint_metadata(path):
    """Exact-resume preflight, before mutating the live model or optimizer."""
    validate_checkpoint(path)
    with zipfile.ZipFile(path) as zf:
        return _zip_metadata(zf)


def _cleanup_old_checkpoints(save_dir: Path, keep_last_n: int = 1, exclude: Optional[Path] = None):
    """Rotate only this process's successful writes, after replacement validation."""
    zips = sorted((p for p in save_dir.glob("seto_step_*.zip")
                   if p.resolve() in _MANAGED_CHECKPOINTS), key=lambda p: p.stat().st_mtime_ns)
    keep = {p.resolve() for p in zips[-keep_last_n:]}
    if exclude:
        keep.add(exclude.resolve())
    for old in zips:
        if old.resolve() not in keep:
            try:
                old.unlink()
                _MANAGED_CHECKPOINTS.discard(old.resolve())
            except OSError as error:
                warnings.warn(f"Checkpoint saved but retention failed for {old}: {error}")


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    directory = Path(save_dir)
    for path in sorted(directory.glob("seto_step_*.zip"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            validate_checkpoint(path)
            return str(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    # Older directory checkpoints are loadable explicitly; require metadata and
    # both states for auto-discovery. Never delete them as part of ZIP retention.
    for path in sorted(directory.glob("step_*"), reverse=True):
        if path.is_dir() and all((path / n).is_file() for n in ("model.pt", "optimizer.pt", "meta.json")):
            try:
                meta = json.loads((path / "meta.json").read_text())
                if type(meta.get("step")) is int and isinstance(meta.get("config"), dict):
                    return str(path)
            except (OSError, ValueError):
                continue
    return None


def clean_checkpoints(save_dir: str):
    """Remove all checkpoint zips and legacy dirs."""
    save_dir = Path(save_dir)
    for f in save_dir.glob("seto_step_*.zip"):
        f.unlink(missing_ok=True)
    for d in save_dir.glob("step_*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def zip_checkpoint(ckpt_dir: str, output_path: str) -> str:
    """ZIP a plain directory for export. Returns path to zip."""
    ckpt_dir = Path(ckpt_dir)
    output_path = Path(output_path)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        for fp in ckpt_dir.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(ckpt_dir))

    return str(output_path)
