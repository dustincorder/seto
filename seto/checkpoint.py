"""Seto checkpoint management — streaming ZIP, no intermediate directories.

Peak disk per checkpoint: single ZIP file (~5.5GB for small).
No directory + ZIP duplication.
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


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
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"step_{step:08d}"
    zip_path = save_dir / f"seto_{ckpt_name}.zip"
    tmp_path = save_dir / f"seto_{ckpt_name}.tmp"

    # Unwrap DDP
    state_dict = model.state_dict()
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()

    # Low-disk: remove old checkpoints BEFORE writing new one
    _cleanup_old_checkpoints(save_dir, keep_last_n, exclude=tmp_path)

    # Write everything directly into ZIP via tmp file
    # (atomic-ish: if crash mid-write, .tmp won't match seto_step_*.zip glob)
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
        with zf.open("model.pt", "w") as f:
            torch.save(state_dict, f)

        # Optimizer state can exceed ZIP's 4 GiB per-entry limit.
        with zf.open("optimizer.pt", "w", force_zip64=True) as f:
            torch.save(optimizer.state_dict(), f)

        if scheduler is not None:
            with zf.open("scheduler.pt", "w") as f:
                torch.save(scheduler.state_dict(), f)

        if scaler is not None and hasattr(scaler, "state_dict"):
            with zf.open("scaler.pt", "w") as f:
                torch.save(scaler.state_dict(), f)

        if rng_state is not None:
            with zf.open("rng.pt", "w") as f:
                torch.save(rng_state, f)

        meta = {
            "step": step,
            "loss": loss,
            "tokens_seen": tokens_seen,
            "config": config,
        }
        zf.writestr("meta.json", json.dumps(meta, indent=2))

    # Atomic rename: .tmp → final name
    tmp_path.rename(zip_path)

    return str(zip_path)


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
        is_distributed = torch.distributed.is_initialized()
        is_main = (not is_distributed) or torch.distributed.get_rank() == 0

        with zipfile.ZipFile(path, "r") as zf:
            model_entry = _find_zip_entry(zf, "model.pt")
            if not model_entry:
                raise FileNotFoundError(f"model.pt not found in zip: {path}")
            # Load model
            with zf.open(model_entry) as f:
                state_dict = _safe_torch_load(f, map_location=device)
            raw_model = model.module if hasattr(model, "module") else model
            _load_model_state(raw_model, state_dict, allow_vocab_growth)

            # Load optimizer
            opt_entry = _find_zip_entry(zf, "optimizer.pt")
            if optimizer is not None and opt_entry:
                with zf.open(opt_entry) as f:
                    optimizer.load_state_dict(
                        _safe_torch_load(f, map_location=device)
                    )

            # Load metadata
            meta = {}
            meta_entry = _find_zip_entry(zf, "meta.json")
            if meta_entry:
                meta = json.loads(zf.read(meta_entry).decode())

            sched_entry = _find_zip_entry(zf, "scheduler.pt")
            if sched_entry:
                with zf.open(sched_entry) as f:
                    meta["scheduler"] = _safe_torch_load(f, map_location=device)

            scaler_entry = _find_zip_entry(zf, "scaler.pt")
            if scaler_entry:
                with zf.open(scaler_entry) as f:
                    meta["scaler"] = _safe_torch_load(f, map_location=device)

            rng_entry = _find_zip_entry(zf, "rng.pt")
            if rng_entry:
                with zf.open(rng_entry) as f:
                    meta["rng"] = _safe_torch_load(f, map_location=device)

        return meta

    # Legacy: plain directory
    state_dict = _safe_torch_load(path / "model.pt", map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    _load_model_state(raw_model, state_dict, allow_vocab_growth)

    if optimizer is not None and (path / "optimizer.pt").exists():
        optimizer.load_state_dict(
            _safe_torch_load(path / "optimizer.pt", map_location=device)
        )

    meta = {}
    if (path / "meta.json").exists():
        with open(path / "meta.json") as f:
            meta = json.load(f)

    if (path / "scheduler.pt").exists():
        meta["scheduler"] = _safe_torch_load(path / "scheduler.pt", map_location=device)

    if (path / "scaler.pt").exists():
        meta["scaler"] = _safe_torch_load(path / "scaler.pt", map_location=device)

    if (path / "rng.pt").exists():
        meta["rng"] = _safe_torch_load(path / "rng.pt", map_location=device)

    return meta


def _cleanup_old_checkpoints(save_dir: Path, keep_last_n: int = 1, exclude: Optional[Path] = None):
    """Keep only the N most recent checkpoint zips. Also cleans legacy dirs."""
    zips = sorted(save_dir.glob("seto_step_*.zip"), key=lambda x: x.stat().st_mtime)
    for old in zips[: len(zips) - keep_last_n]:
        if exclude and old == exclude:
            continue
        old.unlink(missing_ok=True)

    # Clean legacy directories
    for d in save_dir.glob("step_*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    save_dir = Path(save_dir)

    # Prefer ZIPs (native format)
    zips = sorted(save_dir.glob("seto_step_*.zip"), key=lambda x: x.stat().st_mtime)
    if zips:
        return str(zips[-1])

    # Fallback to legacy directories
    dirs = sorted(save_dir.glob("step_*"), key=lambda x: x.stat().st_mtime)
    if dirs:
        return str(dirs[-1])

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
