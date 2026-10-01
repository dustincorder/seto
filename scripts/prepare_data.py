#!/usr/bin/env python3
"""Seto data preparation — download, tokenize, pack into uint16 shards."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto.tokenizer import SetoTokenizer
from seto.data import pack_from_hf_dataset


def prepare_tokenizer(
    output_dir: str,
    vocab_size: int = 48000,
    samples_ru: int = 50000,
    samples_en: int = 40000,
    samples_uk: int = 10000,
    samples_code: int = 5000,
):
    """Train tokenizer on multilingual data (RU/EN/UK/Code)."""
    print(f"Training tokenizer (vocab={vocab_size}, target samples: RU={samples_ru:,}, EN={samples_en:,}, UK={samples_uk:,}, Code={samples_code:,})...")

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

    # English from FineWeb
    if samples_en > 0:
        print(f"  Sampling English text ({samples_en:,} docs)...", flush=True)
        try:
            en_ds = load_dataset("HuggingFaceFW/fineweb_100BT", split="train", streaming=True)
            for i, row in enumerate(en_ds):
                if i >= samples_en:
                    break
                text = row.get("text", "")
                if text and len(text) > 100:
                    samples.append(text)
                if (i + 1) % 10000 == 0:
                    print(f"    Sampled {i + 1:,} EN docs...", flush=True)
        except Exception as e:
            print(f"  Warning: Could not load FineWeb EN: {e}")

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
            code_ds = load_dataset("codeparrot/github-code", name="Python", split="train", streaming=True)
            for i, row in enumerate(code_ds):
                if i >= samples_code:
                    break
                code = row.get("code", "")
                if code and len(code) > 50:
                    samples.append(code)
        except Exception as e:
            print(f"  Warning: Could not load code dataset for tokenizer: {e}")

    if not samples:
        print("ERROR: No samples collected for tokenizer training")
        return

    print(f"  Total samples collected for tokenizer: {len(samples):,}")

    # Write samples to temp file
    temp_file = os.path.join(output_dir, "tokenizer_samples.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(samples))

    # Train tokenizer
    tokenizer = SetoTokenizer(vocab_size=vocab_size)
    tokenizer.train([temp_file], output_dir)

    # Cleanup
    os.remove(temp_file)

    print(f"Tokenizer trained: {len(tokenizer)} vocab")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare Seto training data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--tokenizer-dir", default="seto-tokenizer", help="Tokenizer output dir")
    parser.add_argument("--vocab-size", type=int, default=48000)
    parser.add_argument("--tokenizer-samples-ru", type=int, default=50000)
    parser.add_argument("--tokenizer-samples-en", type=int, default=40000)
    parser.add_argument("--tokenizer-samples-uk", type=int, default=10000)
    parser.add_argument("--tokenizer-samples-code", type=int, default=5000)
    parser.add_argument("--max-samples-ru", type=int, default=500000,
                        help="Max Russian samples from FineWeb2")
    parser.add_argument("--max-samples-en", type=int, default=450000,
                        help="Max English samples from FineWeb2 (0 disables)")
    parser.add_argument("--max-samples-uk", type=int, default=80000,
                        help="Max Ukrainian samples from FineWeb2 (0 disables)")
    parser.add_argument("--max-samples-wiki", type=int, default=30000,
                        help="Max Wikipedia samples")
    parser.add_argument("--max-samples-technical", type=int, default=30000,
                        help="Max technical/code samples (0 disables)")
    parser.add_argument("--shard-size", type=int, default=100_000_000,
                        help="Tokens per shard")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    shard_dir = os.path.join(args.output_dir, "shards")

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

    def next_shard_index():
        return len(list(Path(shard_dir).glob("train_*.bin")))

    # Sources are appended into one shard sequence. Sample counts are approximate
    # token weights because document lengths vary by source.
    print("Packing FineWeb2 Russian...")
    fw_shards = pack_from_hf_dataset(
        "HuggingFaceFW/fineweb-2",
        tokenizer, shard_dir,
        text_key="text",
        max_samples=args.max_samples_ru,
        shard_size=args.shard_size,
        split="train",
        config_name="rus_Cyrl",
        shard_start_idx=0,
    )

    if args.max_samples_en:
        print("Packing FineWeb English...")
        pack_from_hf_dataset(
            "HuggingFaceFW/fineweb_100BT",
            tokenizer, shard_dir,
            text_key="text",
            max_samples=args.max_samples_en,
            shard_size=args.shard_size,
            split="train",
            shard_start_idx=next_shard_index(),
        )

    if args.max_samples_uk:
        print("Packing FineWeb2 Ukrainian...")
        pack_from_hf_dataset(
            "HuggingFaceFW/fineweb-2",
            tokenizer, shard_dir,
            text_key="text",
            max_samples=args.max_samples_uk,
            shard_size=args.shard_size,
            split="train",
            config_name="ukr_Cyrl",
            shard_start_idx=next_shard_index(),
        )

    if args.max_samples_technical:
        print("Packing technical/code data...")
        try:
            pack_from_hf_dataset(
                "codeparrot/github-code",
                tokenizer, shard_dir,
                text_key="code",
                max_samples=args.max_samples_technical,
                shard_size=args.shard_size,
                split="train",
                config_name="Python",
                shard_start_idx=next_shard_index(),
            )
        except Exception as error:
            # Streaming code datasets can terminate after writing partial shards.
            # Keep those shards; do not start another network stream during cleanup.
            print(f"Warning: technical/code stream stopped; keeping partial data: {error}")

    if args.max_samples_wiki:
        # Wikipedia adds Russian reference/encyclopedic text after multilingual data.
        wiki_start = next_shard_index()
        print(f"Packing Wikipedia Russian (starting at shard {wiki_start})...")
        pack_from_hf_dataset(
            "wikimedia/wikipedia",
            tokenizer, shard_dir,
            text_key="text",
            max_samples=args.max_samples_wiki,
            shard_size=args.shard_size,
            split="train",
            config_name="20231101.ru",
            shard_start_idx=wiki_start,
        )

    print(f"\nDone! Data ready at {shard_dir}")
    print(f"Tokenizer at {args.tokenizer_dir}")


if __name__ == "__main__":
    main()
