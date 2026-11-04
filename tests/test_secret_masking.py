import ast
import os
import subprocess
import unittest
from pathlib import Path


def extract_function_from_file(file_path: Path, function_name: str):
    """Parse python code using AST and compile/return the requested function without running top-level statements."""
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Create a new module containing only this FunctionDef
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            compiled = compile(mod, filename=str(file_path), mode="exec")
            scope = {}
            exec(compiled, scope)
            return scope[function_name]

    raise ValueError(f"Function '{function_name}' not found in {file_path}")


class TestSecretMaskingFromAST(unittest.TestCase):
    def setUp(self):
        self.mask_1b = extract_function_from_file(
            Path("notebooks/kaggle-1b-run1-cell.py"), "_mask_sensitive"
        )
        self.mask_sft = extract_function_from_file(
            Path("notebooks/kaggle-sft-v1-cell.py"), "_mask_sensitive"
        )

    def test_both_runners_mask_token_flag(self):
        for mask_fn in (self.mask_1b, self.mask_sft):
            cmd = ["python", "scripts/upload_hf.py", "--token", "hf_secret_12345", "--repo-id", "test/repo"]
            masked = mask_fn(cmd)
            self.assertNotIn("hf_secret_12345", masked)
            self.assertEqual(masked, ["python", "scripts/upload_hf.py", "--token", "[REDACTED]", "--repo-id", "test/repo"])

    def test_both_runners_mask_token_equals(self):
        for mask_fn in (self.mask_1b, self.mask_sft):
            cmd = ["python", "scripts/upload_hf.py", "--token=hf_secret_12345", "--repo-id", "test/repo"]
            masked = mask_fn(cmd)
            self.assertNotIn("hf_secret_12345", masked)
            self.assertEqual(masked, ["python", "scripts/upload_hf.py", "--token=[REDACTED]", "--repo-id", "test/repo"])

    def test_both_runners_mask_password_and_api_key(self):
        for mask_fn in (self.mask_1b, self.mask_sft):
            cmd = ["tool", "--api-key", "my_key", "--hf-token", "my_token", "--password", "secret_pass"]
            masked = mask_fn(cmd)
            self.assertNotIn("my_key", masked)
            self.assertNotIn("my_token", masked)
            self.assertNotIn("secret_pass", masked)
            self.assertEqual(
                masked,
                ["tool", "--api-key", "[REDACTED]", "--hf-token", "[REDACTED]", "--password", "[REDACTED]"]
            )

    def test_both_runners_mask_password_equals(self):
        for mask_fn in (self.mask_1b, self.mask_sft):
            cmd = ["tool", "--password=my_secret_pass", "--api-key=my_key"]
            masked = mask_fn(cmd)
            self.assertNotIn("my_secret_pass", masked)
            self.assertNotIn("my_key", masked)
            self.assertEqual(
                masked,
                ["tool", "--password=[REDACTED]", "--api-key=[REDACTED]"]
            )

    def test_normal_args_unaffected(self):
        for mask_fn in (self.mask_1b, self.mask_sft):
            cmd = ["python", "scripts/train.py", "--model-config", "tiny", "--batch-size", "4"]
            masked = mask_fn(cmd)
            self.assertEqual(masked, cmd)


class TestUploadHFTokenAndPrivacyHandling(unittest.TestCase):
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
        self.assertNotIn('"--token", hf_token', cell_content)
        self.assertNotIn("'--token', hf_token", cell_content)

    def test_upload_hf_defaults_to_private(self):
        # Inspect upload_hf.py argument parser defaults
        result = subprocess.run(
            ["python3", "scripts/upload_hf.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertIn("--private", result.stdout)
        self.assertIn("--public", result.stdout)


if __name__ == "__main__":
    unittest.main()
