#!/usr/bin/env python3
"""Standalone script to test Seto model generation without Jupyter kernel pollution."""

import argparse
import json
import shutil
import zipfile
import sys
from pathlib import Path

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from seto.model import SetoLM
from seto.config import ModelConfig, MODEL_SMALL
from seto.tokenizer import SetoTokenizer


def find_model(model_arg: str | None = None) -> Path:
    if model_arg:
        p = Path(model_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model path not found: {model_arg}")

    candidates = [
        Path("/kaggle/working/final_pretrain.zip"),
        Path("/kaggle/working/seto-small-1-full/final_pretrain.zip"),
        Path("/kaggle/working/seto-small-1-smoke/final_pretrain.zip"),
        Path("final_pretrain.zip"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            return c

    for ckpt_dir in [
        Path("/kaggle/working/seto-small-1-full/checkpoints_pretrain"),
        Path("/kaggle/working/seto-small-1-smoke/checkpoints_pretrain"),
        Path("checkpoints_pretrain"),
    ]:
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("seto_step_*.zip"))
            if ckpts:
                return ckpts[-1]

    raise FileNotFoundError("Could not find final_pretrain.zip or any checkpoints!")


def main():
    parser = argparse.ArgumentParser(description="Test Seto model generation")
    parser.add_argument("--model", type=str, default=None, help="Path to model archive (zip)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer dir")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Используем устройство для инференса: {device}")

    archive_path = find_model(args.model)
    print(f"Загрузка весов из: {archive_path.name} ({archive_path.stat().st_size / 1024**2:.1f} MB)")

    # 1. Config
    model_config = None
    with zipfile.ZipFile(archive_path, "r") as zf:
        cfg_names = [n for n in zf.namelist() if n == "config.json" or n.endswith("/config.json")]
        if cfg_names:
            with zf.open(cfg_names[0]) as f:
                model_config = ModelConfig(**json.load(f))

    if model_config is None:
        model_config = MODEL_SMALL

    model_config.dropout = 0.0
    model_config.use_gradient_checkpointing = False

    # 2. Weights
    model = SetoLM(model_config)
    with zipfile.ZipFile(archive_path, "r") as zf:
        model_entries = [n for n in zf.namelist() if n == "model.pt" or n.endswith("/model.pt")]
        if not model_entries:
            raise FileNotFoundError("model.pt not found in zip archive!")

        tmp_model = Path("/tmp/seto_inference_model.pt")
        with zf.open(model_entries[0]) as src, open(tmp_model, "wb") as dst:
            shutil.copyfileobj(src, dst)

        state_dict = torch.load(tmp_model, map_location="cpu", weights_only=False)
        tmp_model.unlink(missing_ok=True)

    if "tok_embeddings.weight" in state_dict:
        loaded_vocab = state_dict["tok_embeddings.weight"].shape[0]
        if loaded_vocab != model_config.vocab_size:
            print(f"Адаптация эмбеддингов: {model_config.vocab_size} -> {loaded_vocab}")
            model.resize_embeddings(loaded_vocab)
            model_config.vocab_size = loaded_vocab

    model.load_state_dict(state_dict)
    del state_dict
    model = model.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.eval()
    print(f"Модель Seto-Small успешно загружена! Параметров: {model.count_parameters():,}")

    # 3. Tokenizer
    tokenizer = None
    tok_candidates = [
        Path(args.tokenizer) if args.tokenizer else None,
        Path("/kaggle/working/seto-tokenizer"),
        Path("seto-tokenizer"),
    ]
    for tc in tok_candidates:
        if tc and (tc / "tokenizer.json").exists():
            tokenizer = SetoTokenizer.from_pretrained(str(tc))
            break

    if tokenizer is None:
        with zipfile.ZipFile(archive_path, "r") as zf:
            tok_entries = [n for n in zf.namelist() if n.endswith("tokenizer.json")]
            if tok_entries:
                tok_json = zf.read(tok_entries[0]).decode("utf-8")
                tokenizer = SetoTokenizer.from_serialized(tok_json)

    if tokenizer is None:
        raise RuntimeError("Не удалось найти или загрузить токенизатор!")
    print(f"Токенизатор готов, размер словаря: {len(tokenizer)}")

    # 4. Generate function
    @torch.no_grad()
    def generate(prompt: str) -> str:
        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        for _ in range(args.max_new_tokens):
            x_cond = x if x.size(1) <= model_config.max_seq_len else x[:, -model_config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :].clone()

            # Penalty
            for token_id in set(x[0].tolist()):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= 1.1
                else:
                    next_token_logits[token_id] *= 1.1

            # Sampling
            if args.temperature > 0:
                next_token_logits = next_token_logits / args.temperature
                if args.top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, args.top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")
                if args.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > args.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token.item() == tokenizer.eos_id:
                break
            x = torch.cat([x, next_token], dim=1)

        return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)

    prompts = [
        "Искусственный интеллект в современном мире",
        "The development of modern science and technology has",
        "Солнечная система состоит из центральной звезды — Солнца — и",
        "def binary_search(arr, target):",
    ]

    print("\n" + "=" * 60)
    print("ТЕСТ ГЕНЕРАЦИИ (INFERENCE TEST)")
    print("=" * 60)

    for i, p in enumerate(prompts, 1):
        output = generate(p)
        print(f"\n[{i}] Промпт: {p}")
        print("-" * 50)
        print(output.strip())
        print("-" * 50)

    print("\nТест генерации завершен!")


if __name__ == "__main__":
    main()
