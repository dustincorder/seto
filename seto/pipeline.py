"""Multilingual data pipeline and tokenizer evaluation utilities for Seto.

Provides:
- Exact-text deduplication and deterministic document filtering
- Deterministic document-level train/validation splitting
- Binary shard packing (train_*.bin and val_*.bin)
- Shard manifest generation with SHA256 hashes and stats
- Tokenizer efficiency evaluation (compression ratio, byte efficiency)
"""

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


def hash_text_exact(text: str) -> str:
    """Return SHA256 hex digest for exact string deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 digest of a binary file in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DocumentDeduplicator:
    """Exact-text in-memory deduplicator tracking document text hashes.

    LIMITATIONS:
    - This implements exact Unicode string matching only (SHA-256).
    - It does NOT perform semantic, MinHash, near-duplicate, or fuzzy deduplication.
    - Minor formatting, whitespace variations, or typo differences are treated as distinct documents.
    """

    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, text: str) -> bool:
        h = hash_text_exact(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def __len__(self) -> int:
        return len(self.seen_hashes)


def deterministic_document_split(
    doc_id: str,
    val_ratio: float = 0.05,
    salt: str = "seto-split-seed-42",
) -> str:
    """Deterministically assign a document to 'train' or 'val' based on hash.

    Guarantees:
    - Independent of iteration order or dataset streaming batch size.
    - Zero duplication across train and validation splits: a document hash
      maps strictly and consistently to either 'train' or 'val'.
    """
    if val_ratio <= 0.0:
        return "train"
    if val_ratio >= 1.0:
        return "val"

    key = f"{salt}:{doc_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Use first 8 hex characters (32-bit unsigned int)
    int_val = int(digest[:8], 16)
    threshold = int(val_ratio * 0xFFFFFFFF)
    return "val" if int_val < threshold else "train"


def default_document_filter(
    text: str,
    min_chars: int = 100,
    max_chars: int = 100_000,
) -> Tuple[bool, Optional[str]]:
    """Basic document quality filtering.

    Rejection reasons:
    - empty_or_too_short: Text length < min_chars
    - too_long: Text length > max_chars
    - excessive_repetition: Ratio of unique lines or characters indicates spam
    """
    if not text or len(text.strip()) < min_chars:
        return False, "too_short"
    if len(text) > max_chars:
        return False, "too_long"

    # Excessive character repetition check (e.g., 'aaaaaaaaa...')
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines and len(lines) > 5:
        unique_line_ratio = len(set(lines)) / len(lines)
        if unique_line_ratio < 0.2:
            return False, "repetitive_lines"

    return True, None


class ShardPacker:
    """Packs token sequences into uint16 binary shard files with manifest tracking."""

    def __init__(
        self,
        output_dir: str,
        split: str,
        shard_size: int = 50_000_000,
        start_shard_idx: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.shard_idx = start_shard_idx

        self.buffer: List[int] = []
        self.total_tokens = 0
        self.total_documents = 0
        self.created_shards: List[Dict] = []

    def add_tokens(self, token_ids: List[int]):
        """Add encoded token IDs to buffer and flush to disk when shard_size reached."""
        if not token_ids:
            return
        self.buffer.extend(token_ids)
        self.total_documents += 1

        while len(self.buffer) >= self.shard_size:
            chunk = self.buffer[: self.shard_size]
            self._write_shard(chunk)
            self.buffer = self.buffer[self.shard_size :]

    def _write_shard(self, token_chunk: List[int]):
        shard_name = f"{self.split}_{self.shard_idx:04d}.bin"
        shard_path = self.output_dir / shard_name
        
        # Write uint16 little-endian binary data
        # Check range: 0 <= token_id < 65536
        for tid in token_chunk:
            if not (0 <= tid < 65536):
                raise ValueError(f"Token ID {tid} exceeds uint16 bounds (0..65535)")

        binary_data = struct.pack(f"<{len(token_chunk)}H", *token_chunk)
        shard_path.write_bytes(binary_data)

        file_size = len(binary_data)
        sha256_hash = hashlib.sha256(binary_data).hexdigest()

        self.created_shards.append({
            "name": shard_name,
            "path": str(shard_path.name),
            "tokens": len(token_chunk),
            "bytes": file_size,
            "sha256": sha256_hash,
        })
        self.total_tokens += len(token_chunk)
        self.shard_idx += 1

    def close(self):
        """Flush remaining buffer as final partial shard if present."""
        if self.buffer:
            self._write_shard(self.buffer)
            self.buffer = []


def evaluate_tokenizer_efficiency(
    tokenizer,
    texts_by_lang: Dict[str, List[str]],
) -> Dict[str, Dict]:
    """Evaluate tokenizer efficiency on multilingual samples without retraining.

    Computes:
    - raw UTF-8 bytes
    - total tokens
    - bytes per token (higher is better compression)
    - characters per token
    """
    results = {}
    for lang, texts in texts_by_lang.items():
        total_bytes = 0
        total_chars = 0
        total_tokens = 0

        for t in texts:
            if not t:
                continue
            b = len(t.encode("utf-8"))
            c = len(t)
            # Encode without BOS/EOS to measure content token efficiency
            ids = tokenizer.encode(t, add_bos=False, add_eos=False)
            total_bytes += b
            total_chars += c
            total_tokens += len(ids)

        bytes_per_token = total_bytes / max(1, total_tokens)
        chars_per_token = total_chars / max(1, total_tokens)

        results[lang] = {
            "samples_evaluated": len(texts),
            "total_bytes": total_bytes,
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "bytes_per_token": round(bytes_per_token, 4),
            "chars_per_token": round(chars_per_token, 4),
        }
    return results
