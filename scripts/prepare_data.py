#!/usr/bin/env python3
"""Seto data preparation — download, filter, tokenize, and pack into train/val uint16 shards."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto.tokenizer import SetoTokenizer
from seto.pipeline import (
    DocumentDeduplicator,
    ShardPacker,
    deterministic_document_split,
    default_document_filter,
    evaluate_tokenizer_efficiency,
)


def prepare_tokenizer(
    output_dir: str,
    vocab_size: int = 48000,
    samples_ru: int = 50000,
    samples_en: int = 40000,
    samples_uk: int = 10000,
    samples_code: int = 5000,
):
    """Train tokenizer on multilingual data (RU/EN/UK/Code)."""
    print(
        f"Training tokenizer (vocab={vocab_size}, target samples: "
        f"RU={samples_ru:,}, EN={samples_en:,}, UK={samples_uk:,}, Code={samples_code:,})..."
    )

    from datasets import load_dataset

    samples = []

    # Russian from FineWeb2
    if samples_ru > 0:
        print(f"  Sampling Russian text ({samples_ru:,} docs)...", flush=True)
        try:
            ru_ds = load_dataset("HuggingFaceFW/fineweb-2", name="rus_Cyrl", split="train", streaming=True)
            for i, row in enumerate(ru_ds):
                if i >= samples_ru:
                    break
                text = row.get("text", "")
                if text and len(text) > 100:
                    samples.append(text)
                if (i + 1) % 10000 == 0:
                    print(f"    Sampled {i + 1:,} RU docs...", flush=True)
        except Exception as e:
            print(f"  Warning: Could not load FineWeb2 RU: {e}")

    # English from FineWeb-Edu or FineWeb
    if samples_en > 0:
        print(f"  Sampling English text ({samples_en:,} docs)...", flush=True)
        try:
            en_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            for i, row in enumerate(en_ds):
                if i >= samples_en:
                    break
                text = row.get("text", "")
                if text and len(text) > 100:
                    samples.append(text)
                if (i + 1) % 10000 == 0:
                    print(f"    Sampled {i + 1:,} EN docs...", flush=True)
        except Exception as e:
            print(f"  Warning: Could not load FineWeb-Edu EN: {e}")

    # Ukrainian from FineWeb2
    if samples_uk > 0:
        print(f"  Sampling Ukrainian text ({samples_uk:,} docs)...", flush=True)
        try:
            uk_ds = load_dataset("HuggingFaceFW/fineweb-2", name="ukr_Cyrl", split="train", streaming=True)
            for i, row in enumerate(uk_ds):
                if i >= samples_uk:
                    break
                text = row.get("text", "")
                if text and len(text) > 100:
                    samples.append(text)
                if (i + 1) % 5000 == 0:
                    print(f"    Sampled {i + 1:,} UK docs...", flush=True)
        except Exception as e:
            print(f"  Warning: Could not load FineWeb2 UK: {e}")

    # Code / technical
    if samples_code > 0:
        print(f"  Sampling Code text ({samples_code:,} docs)...", flush=True)
        try:
            code_ds = load_dataset("bigcode/the-stack-smol-xl", data_dir="data/python", split="train", streaming=True)
            for i, row in enumerate(code_ds):
                if i >= samples_code:
                    break
                code = row.get("content", "")
                if code and len(code) > 50:
                    samples.append(code)
        except Exception as e:
            print(f"  Warning: Could not load code dataset for tokenizer: {e}")

    if not samples:
        print("ERROR: No samples collected for tokenizer training")
        return

    print(f"  Total samples collected for tokenizer: {len(samples):,}")

    temp_file = os.path.join(output_dir, "tokenizer_samples.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(samples))

    tokenizer = SetoTokenizer(vocab_size=vocab_size)
    tokenizer.train([temp_file], output_dir)
    os.remove(temp_file)

    print(f"Tokenizer trained: {len(tokenizer)} vocab")
    return tokenizer


def process_multilingual_source(
    source_name: str,
    dataset_iter: Iterable[dict],
    tokenizer: SetoTokenizer,
    train_packer: ShardPacker,
    val_packer: ShardPacker,
    deduplicator: DocumentDeduplicator,
    text_key: str = "text",
    max_samples: Optional[int] = None,
    val_ratio: float = 0.05,
    min_chars: int = 100,
    max_chars: int = 100_000,
    split_salt: str = "seto-split-seed-42",
) -> dict:
    """Stream documents from a dataset, filter, deduplicate, split, and pack into shards."""
    stats = {
        "source": source_name,
        "raw_documents_read": 0,
        "accepted_train_documents": 0,
        "accepted_val_documents": 0,
        "rejected_filter": 0,
        "rejected_duplicate": 0,
        "train_tokens_produced": 0,
        "val_tokens_produced": 0,
    }

    initial_train_tokens = train_packer.total_tokens
    initial_val_tokens = val_packer.total_tokens

    for row in dataset_iter:
        if max_samples and (stats["accepted_train_documents"] + stats["accepted_val_documents"]) >= max_samples:
            break

        stats["raw_documents_read"] += 1
        text = row.get(text_key, "")

        # 1. Document quality filter
        accepted, reason = default_document_filter(text, min_chars=min_chars, max_chars=max_chars)
        if not accepted:
            stats["rejected_filter"] += 1
            continue

        # 2. Exact-text deduplication
        if deduplicator.is_duplicate(text):
            stats["rejected_duplicate"] += 1
            continue

        # 3. Deterministic train/val split
        # Use content hash to guarantee zero cross-split duplication and strict reproducibility
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        split = deterministic_document_split(content_hash, val_ratio=val_ratio, salt=split_salt)

        # 4. Tokenization & Packing
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

        if split == "train":
            train_packer.add_tokens(token_ids)
            stats["accepted_train_documents"] += 1
        else:
            val_packer.add_tokens(token_ids)
            stats["accepted_val_documents"] += 1

        total_accepted = stats["accepted_train_documents"] + stats["accepted_val_documents"]
        if total_accepted % 10000 == 0 and total_accepted > 0:
            print(
                f"  [{source_name}] Processed {stats['raw_documents_read']:,} docs -> "
                f"Accepted: {total_accepted:,} (train: {stats['accepted_train_documents']:,}, "
                f"val: {stats['accepted_val_documents']:,}), Duplicates: {stats['rejected_duplicate']:,}",
                flush=True,
            )

    stats["train_tokens_produced"] = train_packer.total_tokens - initial_train_tokens
    stats["val_tokens_produced"] = val_packer.total_tokens - initial_val_tokens
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare Seto multilingual pretraining data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--tokenizer-dir", default="seto-tokenizer", help="Tokenizer directory")
    parser.add_argument("--vocab-size", type=int, default=48000)
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Deterministic fraction of documents allocated to validation (default: 0.05)")
    parser.add_argument("--shard-size", type=int, default=50_000_000,
                        help="Tokens per uint16 shard (default: 50M tokens ≈ 100MB)")
    parser.add_argument("--split-salt", default="seto-split-seed-42",
                        help="Salt string for deterministic train/val splitting hash")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training")

    # Multilingual sampling targets (approximate doc counts)
    parser.add_argument("--max-samples-ru", type=int, default=500000,
                        help="Max Russian samples from FineWeb2 (0 disables)")
    parser.add_argument("--max-samples-en", type=int, default=450000,
                        help="Max English samples from FineWeb-Edu (0 disables)")
    parser.add_argument("--max-samples-uk", type=int, default=80000,
                        help="Max Ukrainian samples from FineWeb2 (0 disables)")
    parser.add_argument("--max-samples-wiki", type=int, default=30000,
                        help="Max Wikipedia Russian samples (0 disables)")
    parser.add_argument("--max-samples-technical", type=int, default=30000,
                        help="Max Python code samples from The Stack Smol XL (0 disables)")

    # Tokenizer training sample counts
    parser.add_argument("--tokenizer-samples-ru", type=int, default=50000)
    parser.add_argument("--tokenizer-samples-en", type=int, default=40000)
    parser.add_argument("--tokenizer-samples-uk", type=int, default=10000)
    parser.add_argument("--tokenizer-samples-code", type=int, default=5000)

    # Optional tokenizer efficiency evaluation
    parser.add_argument("--eval-tokenizer", action="store_true",
                        help="Run lightweight compression evaluation on sampled texts without retraining")

    args = parser.parse_args()

    shard_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    if not args.skip_tokenizer:
        tokenizer = prepare_tokenizer(
            args.tokenizer_dir,
            vocab_size=args.vocab_size,
            samples_ru=args.tokenizer_samples_ru,
            samples_en=args.tokenizer_samples_en,
            samples_uk=args.tokenizer_samples_uk,
            samples_code=args.tokenizer_samples_code,
        )
    else:
        tokenizer = SetoTokenizer.from_pretrained(args.tokenizer_dir)

    deduplicator = DocumentDeduplicator()
    train_packer = ShardPacker(shard_dir, split="train", shard_size=args.shard_size)
    val_packer = ShardPacker(shard_dir, split="val", shard_size=args.shard_size)

    source_stats = []

    from datasets import load_dataset

    # 1. Russian: FineWeb2 rus_Cyrl
    if args.max_samples_ru > 0:
        print(f"\nProcessing FineWeb2 Russian (target: {args.max_samples_ru:,} docs)...")
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-2", name="rus_Cyrl", split="train", streaming=True)
            stat = process_multilingual_source(
                "fineweb2_ru",
                ds,
                tokenizer,
                train_packer,
                val_packer,
                deduplicator,
                text_key="text",
                max_samples=args.max_samples_ru,
                val_ratio=args.val_ratio,
                split_salt=args.split_salt,
            )
            source_stats.append(stat)
        except Exception as e:
            print(f"Warning: Failed streaming FineWeb2 Russian: {e}")

    # 2. English: FineWeb-Edu sample-10BT
    if args.max_samples_en > 0:
        print(f"\nProcessing FineWeb-Edu English (target: {args.max_samples_en:,} docs)...")
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            stat = process_multilingual_source(
                "fineweb_edu_en",
                ds,
                tokenizer,
                train_packer,
                val_packer,
                deduplicator,
                text_key="text",
                max_samples=args.max_samples_en,
                val_ratio=args.val_ratio,
                split_salt=args.split_salt,
            )
            source_stats.append(stat)
        except Exception as e:
            print(f"Warning: Failed streaming FineWeb-Edu English: {e}")

    # 3. Ukrainian: FineWeb2 ukr_Cyrl
    if args.max_samples_uk > 0:
        print(f"\nProcessing FineWeb2 Ukrainian (target: {args.max_samples_uk:,} docs)...")
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-2", name="ukr_Cyrl", split="train", streaming=True)
            stat = process_multilingual_source(
                "fineweb2_uk",
                ds,
                tokenizer,
                train_packer,
                val_packer,
                deduplicator,
                text_key="text",
                max_samples=args.max_samples_uk,
                val_ratio=args.val_ratio,
                split_salt=args.split_salt,
            )
            source_stats.append(stat)
        except Exception as e:
            print(f"Warning: Failed streaming FineWeb2 Ukrainian: {e}")

    # 4. Technical / Code: The Stack Smol XL Python
    if args.max_samples_technical > 0:
        print(f"\nProcessing Python code (The Stack Smol XL)...")
        try:
            ds = load_dataset("bigcode/the-stack-smol-xl", data_dir="data/python", split="train", streaming=True)
            stat = process_multilingual_source(
                "the_stack_smol_xl_python",
                ds,
                tokenizer,
                train_packer,
                val_packer,
                deduplicator,
                text_key="content",
                max_samples=args.max_samples_technical,
                val_ratio=args.val_ratio,
                split_salt=args.split_salt,
            )
            source_stats.append(stat)
        except Exception as e:
            print(f"Warning: Failed streaming code dataset: {e}")

    # 5. Russian Wikipedia
    if args.max_samples_wiki > 0:
        print(f"\nProcessing Russian Wikipedia...")
        try:
            ds = load_dataset("wikimedia/wikipedia", name="20231101.ru", split="train", streaming=True)
            stat = process_multilingual_source(
                "wikipedia_ru",
                ds,
                tokenizer,
                train_packer,
                val_packer,
                deduplicator,
                text_key="text",
                max_samples=args.max_samples_wiki,
                val_ratio=args.val_ratio,
                split_salt=args.split_salt,
            )
            source_stats.append(stat)
        except Exception as e:
            print(f"Warning: Failed streaming Wikipedia Russian: {e}")

    # Flush all remaining buffers
    train_packer.close()
    val_packer.close()

    # Generate Manifest
    manifest = {
        "format": "seto-shards-v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "directory": args.tokenizer_dir,
        },
        "split_config": {
            "val_ratio": args.val_ratio,
            "split_salt": args.split_salt,
            "deduplication": "exact_sha256",
        },
        "totals": {
            "train_tokens": train_packer.total_tokens,
            "train_documents": train_packer.total_documents,
            "train_shards": len(train_packer.created_shards),
            "val_tokens": val_packer.total_tokens,
            "val_documents": val_packer.total_documents,
            "val_shards": len(val_packer.created_shards),
            "unique_documents_seen": len(deduplicator),
        },
        "sources": source_stats,
        "shards": {
            "train": train_packer.created_shards,
            "val": val_packer.created_shards,
        },
    }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Data pipeline finished!")
    print(f"  Train: {train_packer.total_tokens:,} tokens ({len(train_packer.created_shards)} shards)")
    print(f"  Val:   {val_packer.total_tokens:,} tokens ({len(val_packer.created_shards)} shards)")
    print(f"  Manifest written to: {manifest_path}")

    if args.eval_tokenizer:
        print("\nEvaluating multilingual tokenizer compression efficiency...")
        sample_texts = {
            "ru": ["Привет, мир! Это тестовая строка для оценки эффективности токенизатора."] * 10,
            "en": ["Hello world! This is a test string to evaluate tokenizer compression efficiency."] * 10,
            "uk": ["Привіт, світе! Це тестовий рядок для оцінки ефективності токенізатора."] * 10,
        }
        eff = evaluate_tokenizer_efficiency(tokenizer, sample_texts)
        print(json.dumps(eff, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
