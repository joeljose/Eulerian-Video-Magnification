# Contributing

Thanks for your interest in contributing!

## How to contribute

1. **Open an issue first** — describe the bug or feature you'd like to work on.
2. **Fork the repo** and create a branch from `main`.
3. **Keep PRs small** — one logical change per pull request.
4. **Follow PEP 8** for Python code style.
5. **Test your changes** — run the CLI on `face.mp4` to verify nothing is broken:
   ```bash
   python evm.py -i face.mp4 -o test_output.avi -fl 0.83 -fh 1.0 -a 50
   ```
6. **Open a pull request** against `main` with a clear description of your changes.

## Reporting bugs

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS
