#!/usr/bin/env python3
"""Upload Seto model checkpoints to Hugging Face Hub."""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def find_checkpoint(model_arg: str | None = None) -> Path:
    if model_arg:
        p = Path(model_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model path not found: {model_arg}")

    candidates = [
        Path("/kaggle/working/final_sft.zip"),
        Path("/kaggle/working/seto-sft-v1/final_sft.zip"),
        Path("final_sft.zip"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            return c

    # Search in /kaggle/input
    input_matches = sorted(
        Path("/kaggle/input").rglob("final_sft.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    if input_matches:
        return input_matches[-1]

    # Fallback to pretrain
    pretrain_matches = sorted(
        Path("/kaggle/input").rglob("final_pretrain.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    if pretrain_matches:
        return pretrain_matches[-1]

    raise FileNotFoundError("Could not find final_sft.zip in /kaggle/working, /kaggle/input, or local dir")


def main():
    parser = argparse.ArgumentParser(description="Upload Seto model to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo ID, e.g., 'your-username/seto-small-sft'")
    parser.add_argument("--token", default=None, help="HF write token (or set HF_TOKEN env var)")
    parser.add_argument("--model", default=None, help="Path to final_sft.zip (auto-detected if omitted)")
    parser.add_argument("--private", action="store_true", help="Make HF repository private")
    parser.add_argument("--include-unpacked", action="store_true", default=True,
                        help="Also upload unpacked model.pt and config.json")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Missing Hugging Face token! Pass --token hf_... or set export HF_TOKEN=hf_..."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import HfApi

    archive_path = find_checkpoint(args.model)
    print(f"📦 Найден чекпоинт: {archive_path} ({archive_path.stat().st_size / 1024**2:.1f} MB)")

    api = HfApi(token=token)

    print(f"🚀 Создаем/проверяем репозиторий: {args.repo_id} (private={args.private})...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # 1. Upload archive itself
    print(f"📤 Загрузка {archive_path.name} в {args.repo_id}...")
    api.upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=archive_path.name,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"✅ Архив {archive_path.name} успешно загружен!")

    # 2. Upload unpacked files for easy inspection & direct loading
    if args.include_unpacked and archive_path.suffix == ".zip":
        print("📂 Распаковка и загрузка отдельных файлов (config, weights, tokenizer)...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(tmp_path)

            # Look for model.pt, config.json, tokenizer
            extracted_files = list(tmp_path.rglob("*"))
            for f in extracted_files:
                if f.is_file() and not f.name.startswith("."):
                    rel_name = f.relative_to(tmp_path).as_posix()
                    # Flatten directory structure if needed or keep clean
                    target_repo_path = rel_name.split("/")[-1] if "/" in rel_name and not "tokenizer" in rel_name else rel_name
                    print(f"  -> Загрузка {rel_name} как {target_repo_path}...")
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=target_repo_path,
                        repo_id=args.repo_id,
                        repo_type="model",
                    )

    # 3. Create a clean README.md
    readme_content = f"""---
license: apache-2.0
language:
- ru
- en
- uk
tags:
- seto
- pytorch
- causal-lm
- sft
pipeline_tag: text-generation
---

# Seto-Small (490M) — SFT v1

**Seto** — легкая двуязычная модель (русский + английский, с поддержкой украинского), оптимизированная для мобильных устройств и локального инференса.

## Характеристики
- **Параметры:** 490,108,160 (~490M)
- **Словарь (Vocab):** 48,000 токенов (BPE)
- **Архитектура:** Llama-подобная (RoPE, RMSNorm, SwiGLU, Grouped Query Attention)
- **Стадия:** SFT v1 (3000 шагов на миксе UltraChat, WildChat, OASST1, Russian Dialogues, Hermes Tool Calling)
- **Финальный SFT Loss:** ~2.32
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"\n🎉 Все файлы успешно загружены на Hugging Face!")
    print(f"🔗 Ссылка: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
