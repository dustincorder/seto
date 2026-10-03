# Seto-1.1B (1,073,047,552 params) — Pretraining Cell for Kaggle (2x T4 GPUs)
# Run in a clean Kaggle notebook with GPU T4 x2 accelerator.

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch

WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content")
REPO = WORK / "seto"
HF_CACHE = WORK / "seto-hf-cache"
TOKENIZER = WORK / "seto-tokenizer"
DATA = WORK / "seto-1b-data"
OUTPUT = WORK / "seto-1b-pretrain"
DATA_READY = DATA / ".ready"

GPU_COUNT = max(1, torch.cuda.device_count())
BATCH_SIZE = 2      # 2 per GPU to guarantee zero OOM on 16GB T4 with 1.1B params
GRAD_ACCUM = 8      # Effective batch size = 2 * 8 * 2 GPUs = 32 (32,768 tokens per step)
SEQ_LEN = 1024
SAVE_EVERY = 500

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"


def run(*args, cwd=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


# 1. Clone / Sync Seto repository
print("🚀 Синхронизация репозитория Seto...")
if (REPO / ".git").exists():
    run("git", "pull", "--ff-only", cwd=REPO)
else:
    if REPO.exists():
        shutil.rmtree(REPO)
    run("git", "clone", "https://github.com/dustincorder/seto.git", REPO)

run(sys.executable, "-m", "pip", "install", "-q", "tokenizers", "datasets")
sys.path.insert(0, str(REPO))

# 2. Check / Train multilingual tokenizer with <think> tokens
print("\n📦 Подготовка токенизатора Seto (Vocab: 48,000 + <think> tokens)...")
TOKENIZER.mkdir(parents=True, exist_ok=True)
if not (TOKENIZER / "tokenizer.json").exists():
    from seto.tokenizer import SetoTokenizer
    tok = SetoTokenizer(vocab_size=48000)
    # Check if pre-existing tokenizer exists in input
    tok_candidates = list(Path("/kaggle/input").rglob("tokenizer.json"))
    if tok_candidates:
        print(f"Используем готовый токенизатор из: {tok_candidates[0]}")
        shutil.copy2(tok_candidates[0], TOKENIZER / "tokenizer.json")
        cfg_candidates = list(tok_candidates[0].parent.glob("config.json"))
        if cfg_candidates:
            shutil.copy2(cfg_candidates[0], TOKENIZER / "config.json")

# 3. Data packing: Russian + English + Wiki + Python code
DATA.mkdir(parents=True, exist_ok=True)


def shard_token_count():
    return sum(
        path.stat().st_size // np.dtype(np.uint16).itemsize
        for path in sorted((DATA / "shards").glob("*.bin"))
    )


print("\n📚 Проверка обучающих шардов данных...")
existing_tokens = shard_token_count()
if existing_tokens >= 50_000_000 and (TOKENIZER / "tokenizer.json").exists():
    DATA_READY.touch()
    print(f"✅ Используем существующие шарды: {existing_tokens:,} токенов.")
elif not DATA_READY.exists():
    # 250k RU + 250k EN + 30k Wiki + 20k Code ~= 400-500 млн качественных токенов
    pack_cmd = [
        sys.executable, "-u", "scripts/prepare_data.py",
        "--output-dir", DATA,
        "--tokenizer-dir", TOKENIZER,
        "--tokenizer-samples-ru", "40000",
        "--tokenizer-samples-en", "35000",
        "--tokenizer-samples-uk", "8000",
        "--tokenizer-samples-code", "5000",
        "--max-samples-ru", "200000",
        "--max-samples-en", "200000",
        "--max-samples-uk", "30000",
        "--max-samples-wiki", "25000",
        "--max-samples-technical", "20000",
        "--shard-size", "100000000",
    ]
    if (TOKENIZER / "tokenizer.json").exists():
        pack_cmd.append("--skip-tokenizer")

    try:
        run(*pack_cmd, cwd=REPO)
    except subprocess.CalledProcessError as error:
        recovered = shard_token_count()
        if recovered < 20_000_000:
            raise RuntimeError(f"Слишком мало токенов скачано: {recovered:,}") from error
        print(f"Поток частично прерван, но сохранено {recovered:,} токенов. Продолжаем обучение.")

    DATA_READY.touch()

shards = sorted((DATA / "shards").glob("*.bin"))
total_tokens = shard_token_count()
tokens_per_step = BATCH_SIZE * GRAD_ACCUM * GPU_COUNT * SEQ_LEN
max_steps = max(500, total_tokens // tokens_per_step)

print("\n" + "=" * 60)
print(f"ПАРАМЕТРЫ ПРЕТРЕЙНА SETO-1.1B:")
print(f"  GPU:                {GPU_COUNT} x {torch.cuda.get_device_name(0)}")
print(f"  Архитектура:        Seto-1.1B (2048 dim, 22 layers, 16 heads, 4 KV)")
print(f"  Шардов:             {len(shards)}")
print(f"  Всего токенов:      {total_tokens:,}")
print(f"  Токенов на шаг:     {tokens_per_step:,}")
print(f"  Шагов на 1 эпоху:   {max_steps:,}")
print("=" * 60 + "\n")

# 4. Launch Distributed Data Parallel training for Seto-1.1B
OUTPUT.mkdir(parents=True, exist_ok=True)
train_cmd = [
    "torchrun", "--standalone", f"--nproc_per_node={GPU_COUNT}",
    "scripts/train.py",
    "--stage", "pretrain",
    "--model-config", "1b",
    "--data-dir", DATA / "shards",
    "--output-dir", OUTPUT,
    "--tokenizer", TOKENIZER,
    "--batch-size", str(BATCH_SIZE),
    "--grad-accum", str(GRAD_ACCUM),
    "--seq-len", str(SEQ_LEN),
    "--lr", "2.5e-4",
    "--warmup-steps", "500",
    "--max-steps", str(max_steps),
    "--epochs", "1",
    "--save-every", str(SAVE_EVERY),
    "--log-every", "10",
    "--fp16",
    "--gradient-checkpointing",
]

print("🚀 Запуск pretrain команды:")
print(" ".join(map(str, train_cmd)))
run(*train_cmd, cwd=REPO)

# 5. Verify & export final archive
final_zip = OUTPUT / "final_pretrain.zip"
if not final_zip.exists():
    raise FileNotFoundError(f"Обучение завершилось, но {final_zip} не найден!")

root_export = WORK / "final_pretrain_1b.zip"
shutil.copy2(final_zip, root_export)

print("\n" + "=" * 60)
print(f"🎉 ПРЕТРЕЙН SETO-1.1B УСПЕШНО ЗАВЕРШЕН!")
print(f"  Финальный архив:   {final_zip} ({final_zip.stat().st_size / 1024**2:.1f} MB)")
print(f"  Экспорт в корень:  {root_export}")
print("=" * 60)
