# ==============================================================================
# Seto-1.1B: Pretraining Session 1 (Kaggle 2x T4)
# Run in a clean Kaggle notebook with GPU T4 x2 (Save Version / Run All)
# ==============================================================================

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Paths
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content")
REPO = WORK / "seto"
HF_CACHE = WORK / "hf-cache"
TOKENIZER_DIR = WORK / "seto-tokenizer"
DATA_DIR = WORK / "seto-1b-data"
OUTPUT_DIR = WORK / "seto-1b-pretrain"
SHARDS_DIR = DATA_DIR / "shards"

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"

def run(*command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)

# 1. Clone / Pull Repository
if (REPO / ".git").exists():
    run("git", "pull", "--ff-only", cwd=REPO)
else:
    if REPO.exists():
        shutil.rmtree(REPO)
    run("git", "clone", "https://github.com/dustincorder/seto.git", REPO)

sys.path.insert(0, str(REPO))

# 2. Dependencies
run(sys.executable, "-m", "pip", "install", "-q", "tokenizers", "datasets", "huggingface_hub")

# 3. Check if shards and tokenizer already exist (e.g. from previous run attached as Input)
input_shards = sorted(Path("/kaggle/input").rglob("train_0000.bin"))
if input_shards:
    SHARDS_DIR = input_shards[0].parent
    print(f"⚡ Найдена готовая папка с шардами в инпутах: {SHARDS_DIR}")
    tok_candidates = [p.parent for p in Path("/kaggle/input").rglob("tokenizer.json") if "checkpoints" not in p.parts]
    if tok_candidates:
        TOKENIZER_DIR = tok_candidates[0]
        print(f"⚡ Найден готовый токенизатор в инпутах: {TOKENIZER_DIR}")
    DATA_READY = DATA_DIR / ".ready"
    DATA_READY.touch()

DATA_READY = DATA_DIR / ".ready"
if not DATA_READY.exists() and not list(SHARDS_DIR.glob("train_*.bin")):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("📦 Подготовка обучающих данных и токенизатора (с токенами <think>)...")
    run(
        sys.executable,
        "scripts/prepare_data.py",
        "--output-dir", DATA_DIR,
        "--tokenizer-dir", TOKENIZER_DIR,
        "--vocab-size", "48000",
        "--max-samples-ru", "200000",      # FineWeb2 Russian
        "--max-samples-wiki", "30000",     # Russian Wikipedia
        "--max-samples-en", "150000",      # FineWeb English
        "--max-samples-uk", "30000",       # Ukrainian
        "--max-samples-technical", "10000",# Everyday logic / light technical
        "--shard-size", "50000000",        # 50M tokens per shard for fast streaming
        cwd=REPO,
    )
    # Clean raw HF cache to free disk
    shutil.rmtree(HF_CACHE, ignore_errors=True)
    DATA_READY.touch()
    print("✅ Данные и токенизатор готовы!")

# Count available shards
shards = sorted(SHARDS_DIR.glob("train_*.bin"))
print(f"📊 Найдено {len(shards)} шардов данных в {SHARDS_DIR}")
if not shards:
    raise FileNotFoundError(f"Нет шардов данных в {SHARDS_DIR}!")

# 4. Training Seto-1.1B on 2x T4
# Memory configuration for 1.1B on T4:
# - batch_size = 2 per GPU
# - grad_accum = 8
# -> Effective batch size = 2 * 2 * 8 = 32 (32,768 tokens/step at seq_len 1024)
# - FP16 + Gradient Checkpointing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("🚀 ЗАПУСК PRETRAINING SETO-1.1B (2x T4 DDP)")
print("=" * 60)

run(
    "torchrun", "--standalone", "--nproc_per_node=2",
    "scripts/train.py",
    "--stage", "pretrain",
    "--model-config", "1b",
    "--data-dir", SHARDS_DIR,
    "--tokenizer", TOKENIZER_DIR,
    "--output-dir", OUTPUT_DIR,
    "--batch-size", "2",
    "--grad-accum", "8",
    "--seq-len", "1024",
    "--lr", "3e-4",
    "--min-lr", "3e-5",
    "--warmup-steps", "300",
    "--max-steps", "10000",       # Автоматически сохраняет чекпойнты
    "--save-every", "500",
    "--log-every", "10",
    "--fp16",
    "--gradient-checkpointing",
    cwd=REPO,
)

# 5. Pack final pretrain archive
final_zip = OUTPUT_DIR / "final_pretrain.zip"
root_final_zip = WORK / "final_pretrain.zip"

if final_zip.exists():
    shutil.copy2(final_zip, root_final_zip)
    print(f"\n✅ Финальный архив Seto-1.1B готов: {root_final_zip} ({root_final_zip.stat().st_size / 1024**2:.1f} MB)")

# 6. Test Inference
print("\n" + "=" * 60)
print("🧪 ПРОВЕРКА ГЕНЕРАЦИИ SETO-1.1B")
print("=" * 60)
try:
    run(
        sys.executable,
        "scripts/test_inference.py",
        "--model", root_final_zip,
        "--tokenizer", TOKENIZER_DIR,
        cwd=REPO,
    )
except Exception as e:
    print(f"⚠️ Ошибка теста: {e}")

# 7. Optional: Auto-backup to Hugging Face if HF_TOKEN is present
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("\n☁️ Автоматическая выгрузка чекпоинта на Hugging Face...")
    try:
        run(
            sys.executable,
            "scripts/upload_hf.py",
            "--model", root_final_zip,
            "--repo-id", "dustincorder/seto-1.1b-pretrain",
            "--token", hf_token,
            cwd=REPO,
        )
    except Exception as e:
        print(f"⚠️ Ошибка загрузки на HF: {e}")

print("\n🎉 Сессия 1 завершена! Чекпоинт сохранен в Output.")
