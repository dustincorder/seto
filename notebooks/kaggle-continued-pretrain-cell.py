# Paste into one Kaggle notebook cell.
# Continues pretraining from final_pretrain.zip with a fresh optimizer.

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


INPUT_ZIP = Path(
    "/kaggle/input/notebooks/alleydick/seto-0-3-4-pretrain/"
    "seto/seto-small/final_pretrain.zip"
)
WORK = Path("/kaggle/working")
REPO = WORK / "seto"
HF_CACHE = WORK / "hf-pretrain-cache"
BASE_MODEL = WORK / "seto-continued-base"
TOKENIZER = WORK / "seto-continued-tokenizer"
DATA = WORK / "seto-pretrain-data"
OUTPUT = WORK / "seto-continued-pretrain"

# Training budget. Dataset is larger than one night's budget; resume later if needed.
MAX_STEPS = 10_000
SAVE_EVERY = 1_000
MAX_SAMPLES_RU = 800_000
MAX_SAMPLES_WIKI = 100_000


def run(*command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


if not INPUT_ZIP.is_file():
    raise FileNotFoundError(INPUT_ZIP)

# Kaggle input is read-only. The old input checkpoint files are deliberately never copied.
print(f"Using model weights: {INPUT_ZIP}", flush=True)
print(
    "Ignoring old input checkpoints: "
    "/kaggle/input/notebooks/alleydick/seto-0-3-4-pretrain/"
    "seto/seto-small/checkpoints_pretrain/step_0000*",
    flush=True,
)

for path in (REPO, HF_CACHE, BASE_MODEL, TOKENIZER, DATA, OUTPUT):
    if path == OUTPUT:
        continue
    if path.exists():
        shutil.rmtree(path)

existing_checkpoints = sorted(
    (OUTPUT / "checkpoints_pretrain").glob("seto_step_*.zip")
    if (OUTPUT / "checkpoints_pretrain").exists()
    else [],
    key=lambda path: path.stat().st_mtime,
)
resume_checkpoint = existing_checkpoints[-1] if existing_checkpoints else None
if resume_checkpoint:
    print(f"Resuming existing working checkpoint: {resume_checkpoint}", flush=True)

if REPO.exists():
    run("git", "pull", "--ff-only", cwd=REPO)
else:
    run("git", "clone", "https://github.com/mosshaven/seto.git", REPO)

run(sys.executable, "-m", "pip", "install", "-q", "tokenizers", "datasets")
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")


def copy_zip_member(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)


BASE_MODEL.mkdir(parents=True)
tokenizer_members = {}
with zipfile.ZipFile(INPUT_ZIP) as archive:
    names = archive.namelist()

    model_matches = [
        name for name in names
        if name.endswith("/model.pt") and "/tokenizer/" not in name
    ]
    config_matches = [
        name for name in names
        if name.endswith("/config.json") and "/tokenizer/" not in name
    ]
    if len(model_matches) != 1 or len(config_matches) != 1:
        raise RuntimeError(
            f"Expected one model/config in ZIP, found {model_matches}, {config_matches}"
        )

    copy_zip_member(archive, model_matches[0], BASE_MODEL / "model.pt")
    copy_zip_member(archive, config_matches[0], BASE_MODEL / "config.json")

    for filename in ("tokenizer.json", "config.json"):
        matches = [name for name in names if name.endswith("/tokenizer/" + filename)]
        if len(matches) == 1:
            tokenizer_members[filename] = matches[0]
    if "tokenizer.json" not in tokenizer_members:
        raise FileNotFoundError("Tokenizer is missing from final_pretrain.zip")
    for filename, member in tokenizer_members.items():
        copy_zip_member(archive, member, TOKENIZER / filename)

if not (TOKENIZER / "tokenizer.json").exists():
    # Some older final_pretrain exports omit tokenizer files from the ZIP.
    candidates = [
        path
        for path in INPUT_ZIP.parent.rglob("tokenizer.json")
        if "checkpoints" not in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            "Tokenizer missing from final_pretrain.zip and adjacent Kaggle Input"
        )
    tokenizer_json = max(candidates, key=lambda path: path.stat().st_mtime)
    TOKENIZER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_json, TOKENIZER / "tokenizer.json")
    tokenizer_config = tokenizer_json.with_name("config.json")
    if tokenizer_config.exists():
        shutil.copy2(tokenizer_config, TOKENIZER / "config.json")
    print(f"Tokenizer loaded from Input: {tokenizer_json}", flush=True)

print(f"Base model: {BASE_MODEL}", flush=True)
print(f"Tokenizer: {TOKENIZER}", flush=True)

# Reuse the old tokenizer; changing vocabulary would invalidate the base weights.
run(
    sys.executable,
    "scripts/prepare_data.py",
    "--output-dir", DATA,
    "--tokenizer-dir", TOKENIZER,
    "--skip-tokenizer",
    "--max-samples-ru", MAX_SAMPLES_RU,
    "--max-samples-wiki", MAX_SAMPLES_WIKI,
    "--shard-size", "100000000",
    cwd=REPO,
)

train_command = [
    "torchrun", "--standalone", "--nproc_per_node=2",
    "scripts/train.py",
    "--stage", "pretrain",
    "--model-config", "small",
    "--model-config-file", BASE_MODEL / "config.json",
    "--data-dir", DATA / "shards",
    "--tokenizer", TOKENIZER,
    "--output-dir", OUTPUT,
    "--batch-size", "8",
    "--grad-accum", "4",
    "--seq-len", "1024",
    "--lr", "1e-4",
    "--warmup-steps", "200",
    "--max-steps", MAX_STEPS,
    "--save-every", SAVE_EVERY,
    "--log-every", "10",
]
if resume_checkpoint:
    train_command.extend(["--resume", resume_checkpoint])
else:
    train_command.extend(["--init-from", BASE_MODEL])
run(*train_command, cwd=REPO)

final_zip = OUTPUT / "final_pretrain.zip"
if not final_zip.exists():
    raise FileNotFoundError(f"Training finished without {final_zip}")

print(f"Final model: {final_zip}", flush=True)
print(f"Final size: {final_zip.stat().st_size / 1024**3:.2f} GiB", flush=True)
usage = shutil.disk_usage(WORK)
print(f"Working disk free: {usage.free / 1024**3:.2f} GiB", flush=True)
