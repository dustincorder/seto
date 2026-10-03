#!/usr/bin/env python3
"""De-slop and De-refusal filter for Seto SFT datasets.

Removes canned AI refusals ("As an AI...", "Я не могу помочь...") and
corporate flattery/fluff, keeping only direct, substantive knowledge and instructions.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seto.persona import SETO_SYSTEM_PROMPT


# Regex patterns that indicate pure refusal or moralizing boilerplate
REFUSAL_PATTERNS = [
    # Russian refusals
    r"(?i)\bкак (?:искусственный интеллект|ии|языковая модель|нейросеть)\b",
    r"(?i)\bя (?:не могу|не способен|не в силах|не имею возможности) (?:помочь|выполнить|предоставить|написать|ответить)\b",
    r"(?i)\bк сожалению,?\s+я не могу\b",
    r"(?i)\bя всего лишь (?:программа|ии|языковая модель)\b",
    r"(?i)\bизвините,?\s+но я не могу\b",
    r"(?i)\bя не могу найти общий язык\b",
    
    # English refusals
    r"(?i)\bas an? (?:ai|artificial intelligence|language model|large language model)\b",
    r"(?i)\bi (?:cannot|can't|am unable to|am not able to) (?:assist|help|fulfill|provide|generate|write|answer|comply)\b",
    r"(?i)\bi apologize,?\s+but i (?:cannot|can't|am unable)\b",
    r"(?i)\bi'm sorry,?\s+but i (?:cannot|can't|am unable)\b",
    r"(?i)\bit is important to remember that as an? ai\b",
    r"(?i)\bmy safety guidelines (?:prevent|do not allow)\b",

    # Ukrainian refusals
    r"(?i)\bяк (?:штучний інтелект|мовна модель)\b",
    r"(?i)\bя не можу допомогти\b",
]

# Flattery / filler opening patterns to strip
INTRO_SLOP_PATTERNS = [
    r"(?i)^(?:конечно!?|разумеется!?|безусловно!?|с удовольствием!?|с радостью помогу[^.\n]*[.\n]+)",
    r"(?i)^(?:отличный вопрос!?|хороший вопрос!?|интересный вопрос!?\s*)",
    r"(?i)^(?:certainly!?|sure!?|sure thing!?|absolutely!?|i'd be happy to help[^.\n]*[.\n]+)",
    r"(?i)^(?:great question!?|good question!?|interesting question!?\s*)",
    r"(?i)^(?:here is (?:the|a) [^.\n]*:\s*)",
    r"(?i)^(?:вот (?:простой|пошаговый|готовый)? [^.\n]*:\s*)",
]

# Ending filler patterns to strip
OUTRO_SLOP_PATTERNS = [
    r"(?i)(?:надеюсь,?\s+это (?:поможет|было полезно|помогло)[^.\n]*[.\n]*)+$",
    r"(?i)(?:если у вас (?:есть|возникнут|остались) (?:еще|какие-либо)? вопросы[^.\n]*[.\n]*)+$",
    r"(?i)(?:обращайтесь,?\s+если[^.\n]*[.\n]*)+$",
    r"(?i)(?:i hope this helps[^.\n]*[.\n]*)+$",
    r"(?i)(?:let me know if you (?:have|need)[^.\n]*[.\n]*)+$",
    r"(?i)(?:feel free to ask[^.\n]*[.\n]*)+$",
]


def is_refusal(text: str) -> bool:
    """Check if the text contains canned AI refusal language."""
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def clean_slop(text: str) -> str:
    """Strip generic flattery intros and outros."""
    cleaned = text.strip()
    
    # Strip opening fluff
    for pattern in INTRO_SLOP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()
    
    # Strip ending fluff
    for pattern in OUTRO_SLOP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()
        
    return cleaned


def clean_conversation(messages: list[dict]) -> list[dict] | None:
    """Clean a single conversation. Returns cleaned messages or None if rejected."""
    if not messages:
        return None

    cleaned_messages = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if role == "assistant":
            # If the assistant refuses, reject the entire dialogue
            if is_refusal(content):
                return None
            
            # Clean flattery
            content = clean_slop(content)
            if not content or len(content) < 5:
                return None
                
        cleaned_messages.append({"role": role, "content": content})

    # Validate that at least one user and one assistant turn remain
    has_user = any(m["role"] == "user" for m in cleaned_messages)
    has_assistant = any(m["role"] == "assistant" for m in cleaned_messages)
    
    if not (has_user and has_assistant):
        return None

    return cleaned_messages


def main():
    parser = argparse.ArgumentParser(description="Clean SFT dataset from refusals and slop")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output cleaned JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    refusal_dropped = 0
    clean_records = 0

    print(f"🧹 Фильтрация датасета {input_path}...")

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total_records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Standard format: {"messages": [...]} or raw list
            messages = row.get("messages") if isinstance(row, dict) else row
            if not isinstance(messages, list):
                continue

            cleaned = clean_conversation(messages)
            if cleaned is None:
                refusal_dropped += 1
                continue

            clean_records += 1
            fout.write(json.dumps({"messages": cleaned}, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("📊 ИТОГИ ОЧИСТКИ (ZERO-REFUSAL PIPELINE):")
    print(f"  Всего записей на входе:    {total_records:,}")
    print(f"  Вырезано отказов / шлака: {refusal_dropped:,} ({refusal_dropped/max(1, total_records)*100:.1f}%)")
    print(f"  Сохранено чистых записей: {clean_records:,}")
    print(f"  Результат записан в:      {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
