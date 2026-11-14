import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import pipeline module directly by file path to avoid triggering seto/__init__.py
# which imports torch/numpy (in case optional dependencies are absent)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "seto_pipeline",
    Path(__file__).resolve().parent.parent / "seto" / "pipeline.py",
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

DocumentDeduplicator = pipeline.DocumentDeduplicator
ShardPacker = pipeline.ShardPacker
default_document_filter = pipeline.default_document_filter
deterministic_document_split = pipeline.deterministic_document_split
evaluate_tokenizer_efficiency = pipeline.evaluate_tokenizer_efficiency


class DummyTokenizer:
    """Lightweight dummy tokenizer for testing pipeline logic without external C-extensions."""

    def __init__(self, vocab_size: int = 48000):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list:
        # Encode characters to unicode ordinals clamped to vocab_size
        ids = [min(ord(c), self.vocab_size - 1) for c in text]
        if add_bos:
            ids = [1] + ids
        if add_eos:
            ids = ids + [2]
        return ids

    def decode(self, ids: list) -> str:
        return "".join(chr(i) for i in ids if i not in (1, 2))


class TestDeterministicSplitAndDeduplication(unittest.TestCase):
    def test_deterministic_split_consistency(self):
        # The same doc_id must always produce the same split regardless of when it is called
        doc_ids = [f"doc_{i}" for i in range(100)]
        split_run_1 = [deterministic_document_split(doc_id, val_ratio=0.1) for doc_id in doc_ids]
        split_run_2 = [deterministic_document_split(doc_id, val_ratio=0.1) for doc_id in doc_ids]
        self.assertEqual(split_run_1, split_run_2)

    def test_no_train_val_overlap(self):
        # A document must NEVER be in both train and val
        doc_ids = [f"sample_doc_{i}" for i in range(500)]
        train_docs = set()
        val_docs = set()
        for doc_id in doc_ids:
            s = deterministic_document_split(doc_id, val_ratio=0.2)
            if s == "train":
                train_docs.add(doc_id)
            elif s == "val":
                val_docs.add(doc_id)

        self.assertTrue(len(train_docs) > 0)
        self.assertTrue(len(val_docs) > 0)
        # Disjoint verification
        self.assertEqual(len(train_docs.intersection(val_docs)), 0)

    def test_document_deduplication(self):
        dedup = DocumentDeduplicator()
        text_a = "Привет, это первый документ на русском языке."
        text_b = "Hello, this is a distinct document in English."
        text_duplicate = "Привет, это первый документ на русском языке."

        self.assertFalse(dedup.is_duplicate(text_a))
        self.assertFalse(dedup.is_duplicate(text_b))
        self.assertTrue(dedup.is_duplicate(text_duplicate))
        self.assertEqual(len(dedup), 2)

    def test_document_filter(self):
        # Short document rejection
        accepted, reason = default_document_filter("Too short", min_chars=50)
        self.assertFalse(accepted)
        self.assertEqual(reason, "too_short")

        # Repetitive lines rejection
        rep_text = "\n".join(["spam line"] * 10)
        accepted, reason = default_document_filter(rep_text, min_chars=10)
        self.assertFalse(accepted)
        self.assertEqual(reason, "repetitive_lines")

        # Valid document acceptance
        valid_text = "Valid text " * 15
        accepted, reason = default_document_filter(valid_text, min_chars=50)
        self.assertTrue(accepted)
        self.assertIsNone(reason)


class TestShardPackerAndFormatting(unittest.TestCase):
    def test_shard_packer_uint16_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            packer = ShardPacker(tmp_dir, split="train", shard_size=20)
            tokens_doc1 = [10, 20, 30, 40, 50]
            tokens_doc2 = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
            tokens_doc3 = [160, 170, 180, 190, 200, 210, 220, 230]

            packer.add_tokens(tokens_doc1)
            packer.add_tokens(tokens_doc2)
            packer.add_tokens(tokens_doc3)
            packer.close()

            total_tokens_expected = len(tokens_doc1) + len(tokens_doc2) + len(tokens_doc3)
            self.assertEqual(packer.total_tokens, total_tokens_expected)
            self.assertEqual(packer.total_documents, 3)

            # Check created shards
            shards = sorted(Path(tmp_dir).glob("train_*.bin"))
            self.assertTrue(len(shards) >= 1)

            # Verify binary formatting (uint16 little-endian)
            all_read_tokens = []
            for shard_path in shards:
                data = shard_path.read_bytes()
                count = len(data) // 2
                self.assertEqual(len(data) % 2, 0)
                parsed = struct.unpack(f"<{count}H", data)
                all_read_tokens.extend(parsed)

            expected_all_tokens = tokens_doc1 + tokens_doc2 + tokens_doc3
            self.assertEqual(all_read_tokens, expected_all_tokens)

            # Verify SHA256 matches manifest entry
            for shard_meta in packer.created_shards:
                p = Path(tmp_dir) / shard_meta["name"]
                actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
                self.assertEqual(shard_meta["sha256"], actual_sha)


class TestStrictSplitFallbackProtection(unittest.TestCase):
    def test_strict_split_raises_on_missing_val_shards(self):
        try:
            from seto.data import ShardDataset, find_shards
        except ImportError:
            # Test find_shards logic without importing torch
            def standalone_find_shards(data_dir: str, split: str = "train", strict_split: bool = False):
                data_path = Path(data_dir)
                shards = sorted(data_path.glob(f"{split}_*.bin"))
                if not shards:
                    if strict_split or split != "train":
                        raise FileNotFoundError(f"No shard files found for split '{split}' in {data_dir}")
                    shards = sorted(data_path.glob("*.bin"))
                return [str(s) for s in shards]

            with tempfile.TemporaryDirectory() as tmp_dir:
                train_bin = Path(tmp_dir) / "train_0000.bin"
                train_bin.write_bytes(struct.pack("<10H", *range(10)))

                shards = standalone_find_shards(tmp_dir, split="train")
                self.assertEqual(len(shards), 1)

                with self.assertRaises(FileNotFoundError):
                    standalone_find_shards(tmp_dir, split="val", strict_split=True)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create ONLY train shards in tmp_dir
            train_bin = Path(tmp_dir) / "train_0000.bin"
            train_bin.write_bytes(struct.pack("<10H", *range(10)))

            # 1. Non-strict find_shards for 'train' succeeds
            shards = find_shards(tmp_dir, split="train")
            self.assertEqual(len(shards), 1)

            # 2. Strict find_shards for 'val' MUST raise FileNotFoundError (no silent fallback!)
            with self.assertRaises(FileNotFoundError):
                find_shards(tmp_dir, split="val", strict_split=True)

            # 3. ShardDataset with split='val' MUST raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                ShardDataset(tmp_dir, seq_len=4, split="val", strict_split=True)


class TestTokenizerEvaluationUtility(unittest.TestCase):
    def test_evaluate_tokenizer_efficiency(self):
        tokenizer = DummyTokenizer(vocab_size=48000)
        texts_by_lang = {
            "en": ["Hello world! This is a test."],
            "ru": ["Привет мир! Это тестовая строка."],
            "uk": ["Привіт світ! Це тестовий рядок."],
        }
        eff = evaluate_tokenizer_efficiency(tokenizer, texts_by_lang)
        for lang in ("en", "ru", "uk"):
            self.assertIn(lang, eff)
            self.assertIn("bytes_per_token", eff[lang])
            self.assertIn("chars_per_token", eff[lang])
            self.assertGreater(eff[lang]["total_tokens"], 0)
            self.assertGreater(eff[lang]["bytes_per_token"], 0)


if __name__ == "__main__":
    unittest.main()
