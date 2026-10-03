#!/usr/bin/env python3
"""Prepare reasoning & thinking datasets with <think> ... </think> for Seto-1.1B."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.clean_sft import is_refusal, clean_slop


def format_reasoning_example(prompt: str, thoughts: str, answer: str) -> dict | None:
    prompt = prompt.strip()
    thoughts = thoughts.strip()
    answer = answer.strip()

    if not prompt or not answer:
        return None

    if is_refusal(thoughts) or is_refusal(answer):
        return None

    answer = clean_slop(answer)
    if not answer:
        return None

    if thoughts:
        assistant_content = f"<think>\n{thoughts}\n</think>\n{answer}"
    else:
        assistant_content = answer

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def stream_bespoke_stratos(max_samples: int = 15000):
    """Bespoke-Stratos contains step-by-step reasoning traces from high quality models."""
    from datasets import load_dataset
    print("📥 Загрузка Bespoke-Stratos-17k (Reasoning Traces)...")
    try:
        ds = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train", streaming=True)
    except Exception as e:
        print(f"⚠️ Не удалось загрузить Bespoke-Stratos: {e}")
        return

    count = 0
    for row in ds:
        if count >= max_samples:
            break
        conversations = row.get("conversations", [])
        if len(conversations) < 2:
            continue
        
        prompt = ""
        thoughts = row.get("thought", "")
        answer = ""

        for turn in conversations:
            if turn.get("from") == "human" or turn.get("role") == "user":
                prompt = turn.get("value") or turn.get("content") or ""
            elif turn.get("from") == "gpt" or turn.get("role") == "assistant":
                answer = turn.get("value") or turn.get("content") or ""

        if not thoughts and "<think>" in answer:
            # Already formatted
            parts = answer.split("</think>")
            if len(parts) == 2:
                thoughts = parts[0].replace("<think>", "").strip()
                answer = parts[1].strip()

        example = format_reasoning_example(prompt, thoughts, answer)
        if example:
            yield example
            count += 1


def stream_open_r1(max_samples: int = 20000):
    """Open-R1 reasoning math & code problems."""
    from datasets import load_dataset
    print("📥 Загрузка Open-R1 (Math & Code Reasoning)...")
    try:
        ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    except Exception as e:
        print(f"⚠️ Не удалось загрузить OpenR1: {e}")
        return

    count = 0
    for row in ds:
        if count >= max_samples:
            break
        prompt = row.get("problem", "")
        thoughts = row.get("generation", "")
        answer = row.get("solution", "")

        if "<think>" in thoughts and "</think>" in thoughts:
            parts = thoughts.split("</think>")
            thoughts = parts[0].replace("<think>", "").strip()
            if not answer:
                answer = parts[1].strip()

        example = format_reasoning_example(prompt, thoughts, answer)
        if example:
            yield example
            count += 1


def main():
    parser = argparse.ArgumentParser(description="Prepare Reasoning Dataset with <think>")
    parser.add_argument("--output", default="data/reasoning_sft.jsonl", help="Output path")
    parser.add_argument("--max-samples", type=int, default=30000, help="Total samples")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    print(f"🧠 Сборка обучающей выборки с размышлениями (<think>)...")

    with open(output_path, "w", encoding="utf-8") as f:
        # 1. Stratos reasoning
        for ex in stream_bespoke_stratos(max_samples=args.max_samples // 2):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            saved += 1
            if saved % 1000 == 0:
                print(f"  Сохранено {saved:,} примеров...")

        # 2. Open-R1 math & logic
        for ex in stream_open_r1(max_samples=args.max_samples - saved):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            saved += 1
            if saved % 1000 == 0:
                print(f"  Сохранено {saved:,} примеров...")

    print(f"\n🎉 Успешно собрано {saved:,} чистых примеров с рассуждениями в {output_path}!")


if __name__ == "__main__":
    main()
