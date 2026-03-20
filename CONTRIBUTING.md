# Contributing

Thanks for your interest in contributing!

## How to contribute

1. **Open an issue first** — describe the bug or feature you'd like to work on.
2. **Fork the repo** and create a branch from `main`.
3. **Keep PRs small** — one logical change per pull request.
4. **Follow PEP 8** for Python code style. We use [ruff](https://docs.astral.sh/ruff/) for linting.
5. **Test your changes** before opening a PR:
   ```bash
   # CPU changes
   ./test.sh

   # GPU/CUDA changes
   ./test.sh gpu
   ```
   Tests run inside Docker — no local Python dependencies needed. See [Development](README.md#development) in the README for details.
6. **Open a pull request** against `main` with a clear description of your changes.

## Reporting bugs

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS
