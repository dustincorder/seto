"""Reproduce architecture counts without allocating preset weight storage.

Run: python -m scripts.architecture_report
"""

from dataclasses import replace

import torch

from seto.config import MODEL_BASE, MODEL_SMALL, MODEL_TINY
from seto.model import SetoLM


def main():
    print("preset,experiment,head_dim,formula,physical")
    for name, base in [("tiny", MODEL_TINY), ("small", MODEL_SMALL), ("base", MODEL_BASE)]:
        for experiment, config in [
            ("classic", replace(base)),
            ("qk_norm", replace(base, qk_norm=True)),
            ("qk_norm_swa", replace(base, qk_norm=True, sliding_window=256)),
            ("head128_only", replace(base, head_dim=128)),
        ]:
            with torch.device("meta"):
                model = SetoLM(config)
            physical = sum(p.numel() for p in model.parameters())
            assert config.num_params() == physical
            print(f"{name},{experiment},{config.resolved_head_dim},{config.num_params()},{physical}")


if __name__ == "__main__":
    main()
