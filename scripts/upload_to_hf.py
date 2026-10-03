#!/usr/bin/env python3
"""Upload Seto checkpoints or exported models to Hugging Face Hub."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload Seto artifacts to Hugging Face Hub")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to single file (e.g. final_sft.zip or model.gguf)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to folder to upload all contents from",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g. 'username/seto-small-sft')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF Write Token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repository private (default: True, private repos are free on HF)",
    )
    parser.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="Make repository public",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Ошибка: HF токен не найден! Передайте --token hf_... или задайте os.environ['HF_TOKEN'].")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Устанавливаем huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"Проверка/создание репозитория {args.repo} (private={args.private})...")
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"Загрузка файла {file_path.name} ({file_size_mb:.1f} MB) в {args.repo}...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"✅ Файл успешно загружен: https://huggingface.co/{args.repo}")

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Папка не найдена: {dir_path}")
        print(f"Загрузка папки {dir_path} в {args.repo}...")
        api.upload_folder(
            folder_path=str(dir_path),
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"✅ Папка успешно загружена: https://huggingface.co/{args.repo}")
    else:
        print("Укажите --file или --dir для загрузки.")
        sys.exit(1)


if __name__ == "__main__":
    main()
