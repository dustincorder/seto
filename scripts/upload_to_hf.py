#!/usr/bin/env python3
"""Upload Seto checkpoints and artifacts to Hugging Face Hub."""

import argparse
import os
import sys
from pathlib import Path


def upload_to_hub(
    file_or_dir: str,
    repo_id: str,
    token: str | None = None,
    path_in_repo: str | None = None,
    repo_type: str = "model",
    private: bool = False,
):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("huggingface_hub is not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import HfApi, create_repo

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF token not provided. Pass --token or set HF_TOKEN environment variable.")

    api = HfApi(token=token)

    print(f"Ensuring repository exists: {repo_id} ({repo_type}, private={private})...")
    create_repo(repo_id=repo_id, token=token, repo_type=repo_type, private=private, exist_ok=True)

    path = Path(file_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"File or directory not found: {path}")

    if path.is_file():
        target_path = path_in_repo or path.name
        print(f"Uploading file {path} ({path.stat().st_size / 1024**2:.1f} MB) to {repo_id}/{target_path}...")
        url = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Successfully uploaded: {url}")
        return url
    else:
        print(f"Uploading folder {path} to {repo_id}...")
        url = api.upload_folder(
            folder_path=str(path),
            path_in_repo=path_in_repo or "",
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Successfully uploaded folder: {url}")
        return url


def main():
    parser = argparse.ArgumentParser(description="Upload Seto artifacts to Hugging Face Hub")
    parser.add_argument("--path", required=True, help="Path to file or folder (e.g. final_sft.zip)")
    parser.add_argument("--repo-id", required=True, help="HF repository ID (e.g. username/seto-small-sft)")
    parser.add_argument("--token", default=None, help="Hugging Face write token (or set HF_TOKEN env var)")
    parser.add_argument("--path-in-repo", default=None, help="Target path inside repo (defaults to basename)")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset"], help="Repo type")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    upload_to_hub(
        file_or_dir=args.path,
        repo_id=args.repo_id,
        token=args.token,
        path_in_repo=args.path_in_repo,
        repo_type=args.repo_type,
        private=args.private,
    )


if __name__ == "__main__":
    main()
