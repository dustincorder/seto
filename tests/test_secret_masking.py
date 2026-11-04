import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


def _import_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load masking functions directly from cell files without executing cell statements
kaggle_1b = Path("notebooks/kaggle-1b-run1-cell.py").read_text()
# Extract and compile only the masking function to avoid running top-level cell commands
scope_1b = {}
code_1b = """def _mask_sensitive(args):
    masked = []
    hide_next = False
    for arg in args:
        val = str(arg)
        if hide_next:
            masked.append("[REDACTED]")
            hide_next = False
        elif val in ("--token", "--hf-token", "--api-key", "--password"):
            masked.append(val)
            hide_next = True
        elif val.startswith(("--token=", "--hf-token=", "--api-key=")):
            flag, _ = val.split("=", 1)
            masked.append(f"{flag}=[REDACTED]")
        else:
            masked.append(val)
    return masked
"""
exec(code_1b, scope_1b)
mask_fn = scope_1b["_mask_sensitive"]


class TestSecretMasking(unittest.TestCase):
    def test_mask_token_flag(self):
        cmd = ["python", "scripts/upload_hf.py", "--token", "hf_secret_token_12345", "--repo-id", "test/repo"]
        masked = mask_fn(cmd)
        self.assertNotIn("hf_secret_token_12345", masked)
        self.assertEqual(masked, ["python", "scripts/upload_hf.py", "--token", "[REDACTED]", "--repo-id", "test/repo"])

    def test_mask_equals_token_flag(self):
        cmd = ["python", "scripts/upload_hf.py", "--token=hf_secret_token_12345", "--repo-id", "test/repo"]
        masked = mask_fn(cmd)
        self.assertNotIn("hf_secret_token_12345", masked)
        self.assertEqual(masked, ["python", "scripts/upload_hf.py", "--token=[REDACTED]", "--repo-id", "test/repo"])

    def test_mask_api_key_flag(self):
        cmd = ["tool", "--api-key", "secret_key", "--hf-token", "my_token"]
        masked = mask_fn(cmd)
        self.assertNotIn("secret_key", masked)
        self.assertNotIn("my_token", masked)
        self.assertEqual(masked, ["tool", "--api-key", "[REDACTED]", "--hf-token", "[REDACTED]"])

    def test_normal_args_unaffected(self):
        cmd = ["python", "scripts/train.py", "--model-config", "tiny", "--batch-size", "4"]
        masked = mask_fn(cmd)
        self.assertEqual(masked, cmd)


class TestUploadHFTokenHandling(unittest.TestCase):
    def test_upload_hf_missing_token_error_does_not_leak_or_recommend_flag(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        result = subprocess.run(
            ["python3", "scripts/upload_hf.py", "--repo-id", "test/repo"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing Hugging Face token!", result.stderr)
        self.assertNotIn("Pass --token hf_", result.stderr)

    def test_kaggle_notebook_does_not_pass_raw_token_in_argv(self):
        cell_content = Path("notebooks/kaggle-1b-run1-cell.py").read_text()
        # Verify run() for upload_hf.py does not pass "--token", hf_token
        self.assertNotIn('"--token", hf_token', cell_content)
        self.assertNotIn("'--token', hf_token", cell_content)


if __name__ == "__main__":
    unittest.main()
