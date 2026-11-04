# Contributing to Seto

Thank you for your interest in contributing to Seto. As an experimental project exploring tiny language models for edge devices, we welcome improvements to training efficiency, data pipelines, tooling, and documentation.

## Guiding Principles

1. **Lightweight & Focused**: Seto aims for edge-friendly models (up to 2B parameters) trained from scratch. Keep additions lean and avoid heavy external dependencies.
2. **Reproducibility**: Code should be deterministic where possible, with clear configuration schemas and manifests.
3. **Architecture Status**: The underlying model architecture is exploratory and subject to refinement. Major architectural shifts should begin with an issue or discussion before opening large PRs.

## Development Workflow

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/dustincorder/seto.git
cd seto

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 2. Branching & Commits

- Create feature or chore branches off `master` (e.g. `feat/data-filter-speedup` or `chore/docs-update`).
- Write concise, descriptive commit messages adhering to standard conventional commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`).

### 3. Code Standards & Linting

Seto uses `ruff` for linting and code formatting:

```bash
# Check code style
ruff check .

# Format code
ruff format .
```

Ensure syntax and imports compile cleanly:

```bash
python -m py_compile seto/*.py scripts/*.py
```

### 4. Pull Requests

- Keep pull requests focused on a single topic or fix.
- Describe the changes made, motivation, and verification steps performed.
- Ensure temporary files, model weights, and shard caches are not included in the PR.
