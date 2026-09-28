# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- The GPU pyramid used different filters from the CPU one (Gaussian sigma 1 and bilinear zoom instead of OpenCV's 5-tap kernel), so CPU and GPU amplified different content; both now use OpenCV's operators and match to float precision (#35)
- Output pixels were truncated to uint8 instead of rounded, which darkened the output slightly
- `nan` and `inf` were accepted for `-a`, `-fl`, `-fh`, `--lambda-c` and `--fps`
- Per-level gains went negative for small frames with the default `--lambda-c` (for example 320×240 at `-a 10`), so the motion was shrunk instead of magnified; they are now clamped at 0
- Slow drift over the clip turned into amplified flicker because the FFT joined the last frame to the first; the clip is now extended with its time-reversed copy before filtering. Output changes, mostly near the start and end of the clip (#36)
- A failed save no longer prints "Output saved" and exits 0; the output directory is checked before processing (#30)
- Unreadable input, fps of 0 and clips with fewer than 2 frames give a clear error instead of a traceback (#31)
- Frames past an under-reported `CAP_PROP_FRAME_COUNT` are no longer dropped (#31)
- The CUDA tool runs its VRAM check with the decoded frame count and opens the input once (#31)
- A frequency band that contains no FFT bins is now an error instead of silently returning the input (#29)
- `--lambda-c` help and docs described its effect backwards: lower values give stronger amplification (#28)

### Changed
- `evm.py` and `evm_cuda.py` are merged into one implementation: `python evm.py --gpu [--device N]` runs the same code on CuPy arrays. `evm_cuda.py` remains for one release and runs `evm.py --gpu`; the CUDA Docker image runs `evm.py --gpu` (#39)
- The version is kept in `VERSION` and `evm.__version__` only (no more `-cuda` suffix, which SemVer reads as a pre-release) (#39)
- The pyramid is built and collapsed in blocks of 32 frames instead of one frame at a time
- `--freq-high` above the Nyquist frequency is now an error instead of a warning
- The narrow-band warning now fires when the band is narrower than the clip's frequency resolution (`fps / frames`)
- The temporal filter uses a real FFT (`rfft`), which halves its working memory
- ruff and pytest are pinned exactly, with the rule set in `pyproject.toml` (#33)
- The CUDA image is only built in CI when its inputs change (#34)

### Added
- `pip install .` / `pip install .[cuda]` install the `evm` command and module (#39)
- `--fps` to set the input frame rate for files that don't report one
- Synthetic validation: `scripts/synthetic_shapes.py` and `tests/test_synthetic_shapes.py` measure gain, phase lag and isotropy on pulsating shapes with exact ground truth (adapted from Motion-Magnification-Using-2D-DTCWT)
- The effective frequency band, bin count and per-level gains are printed at startup (#28, #29)
- CI runs the unit tests, checks that the output is magnified, and runs `evm_cuda.py` on the CPU through a CuPy shim (#34)

## [2.1.0] - 2026-03-20

### Added
- Unit tests for core CPU functions (color conversion, bandpass filter, pyramid ops)
- GPU unit tests (VRAM estimation, pyramid ops, bandpass filter)
- `VERSION` file as single source of truth for versioning
- Docker image version labels
- `CHANGELOG.md`
- `requirements-dev.txt` for dev dependencies (pytest, ruff)
- `test.sh` for running CPU/GPU tests inside Docker
- Design doc (`docs/design/evm-hardening.md`)
- Development section in README (testing, versioning, project structure)

### Fixed
- `load_video` buffer overflow when `CAP_PROP_FRAME_COUNT` underreports
- `docker-build-cuda.sh` fragile version extraction replaced with simple grep

### Changed
- Dev dependencies (pytest, ruff) baked into Docker images — no runtime installs
- CI streamlined to lint + smoke tests; unit tests run locally via `test.sh`
- Build scripts read version from `VERSION` file instead of grepping Python source
- `cupy-cuda12x` dependency pinned with upper bound (`<14`)
