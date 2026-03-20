# Design Doc: EVM Project Hardening

**Status: APPROVED**

## Context

The Eulerian Video Magnification project is a Python implementation of the MIT CSAIL algorithm (Wu et al., SIGGRAPH 2012) with two variants: CPU (`evm.py` using NumPy/SciPy) and GPU (`evm_cuda.py` using CuPy). Both are standalone CLI scripts shipped via Docker.

A code review identified gaps: no unit tests, no design documentation, fragile versioning, and minor bugs. This design doc covers both the retroactive architecture decisions and the hardening plan.

## Goals and Non-Goals

**Goals:**
- Add unit tests for CPU codepath core functions
- Establish single source of truth for versioning with automated release workflow
- Document non-obvious architectural decisions for future contributors
- Fix `load_video` buffer edge case
- Add changelog, dev dependencies file, and CI test step

**Non-Goals:**
- Refactoring into a pip-installable package
- GPU testing in CI (no hosted GPU runners)
- Extracting shared code into a common module
- Integration tests with video output comparison
- Pinning Docker base images by SHA256 digest

## Proposed Design

### A. Architecture Decisions (retroactive)

**Why YIQ color space?**
The paper specifies YIQ/NTSC because the human visual system is more sensitive to luminance (Y) than chrominance (I, Q). This allows differential amplification — full amplification on Y, attenuated on I/Q via `chrom_attenuation` — reducing color artifacts. The conversion matrices match MATLAB's `rgb2ntsc`/`ntsc2rgb` exactly for reproducibility against the reference implementation.

**Why CuPy for GPU acceleration?**
CuPy mirrors NumPy's API almost 1:1, which minimized the porting effort from `evm.py`. The GPU version is structurally identical to the CPU version — same functions, same flow, just `np` → `cp`. Alternatives considered below.

**Why MJPG codec in AVI container?**
MJPG is available in OpenCV's bundled FFmpeg on all platforms without external codec installation. H.264 would produce smaller files but requires `libx264`, which isn't bundled in `opencv-python` and would add a system dependency. For a research/demo tool, file size is secondary to portability.

**Why OpenCV's pyrDown/pyrUp for pyramid operations?**
The CPU version uses `cv2.pyrDown`/`cv2.pyrUp` because they implement the exact Gaussian pyramid operations from the paper with correct border handling. The GPU version can't use these (no CuPy equivalent), so it approximates with `gaussian_filter` + subsampling (`gpu_pyr_down`) and `zoom` + `gaussian_filter` (`gpu_pyr_up`). This is the main source of numerical difference between CPU and GPU outputs.

**Why VRAM estimation before processing?**
The GPU pipeline allocates the full Laplacian video pyramid on VRAM at once (all levels × all frames). For a 1080p 10-second video at 30fps, this is ~3.5GB. Rather than failing mid-processing with a cryptic CUDA OOM, `check_vram` estimates peak usage upfront and exits with an actionable error message. The estimate accounts for pyramid levels (each ~1/4 previous) plus one FFT buffer at complex64 for the largest filtered level.

### B. Hardening Changes

**Versioning — Approach B (build-time injection):**
- Add `VERSION` file at repo root containing the version string (e.g., `2.1.0`)
- `evm.py` and `evm_cuda.py` keep `__version__` baked in as today
- `/release` skill reads `VERSION`, stamps values into both files (`evm_cuda.py` appends `-cuda`), updates `CHANGELOG.md`, commits, and tags
- Build scripts (`docker-build.sh`, `docker-build-cuda.sh`) read from `VERSION` for image tags
- Dockerfiles receive version as `ARG` for `LABEL version=`
- Scripts remain fully self-contained — no runtime file dependency

**Testing — pytest with tiered tolerances:**
- `requirements-dev.txt` with pytest and ruff
- Test file: `tests/test_evm.py`
- Tier 1 (strict, `atol=1e-6`): `rgb_to_yiq`/`yiq_to_rgb` roundtrip, `estimate_vram_bytes` exact equality
- Tier 2 (moderate, `atol=1e-4`): `ideal_bandpass_filter` (known signal), `collapse_laplacian_pyramid` (build+collapse roundtrip)
- Tier 3 (smoke): `create_laplacian_video_pyramid` (shapes, dtypes, no NaN), input validation error paths
- GPU code: tested locally by developers, CI lints only
- CI: add `pytest` step before existing smoke test

**`load_video` fix:**
- Add `if i >= frame_count: break` guard in both `evm.py` and `evm_cuda.py`
- Handles the edge case where `CAP_PROP_FRAME_COUNT` underreports
- Zero memory overhead vs list+stack approach

**`docker-build-cuda.sh` cleanup:**
- Replace convoluted Python one-liner with `grep -oP` matching CPU script

**Changelog:**
- Create `CHANGELOG.md` with Keep a Changelog format
- Start fresh with `[Unreleased]` section — no backfill (no old tags exist)
- Future releases managed by `/release` skill

## Alternatives Considered

### GPU framework: CuPy vs PyTorch vs Numba vs PyCUDA

| Framework | Pros | Cons | Verdict |
|-----------|------|------|---------|
| CuPy | 1:1 NumPy API, minimal porting, has scipy.fftpack | Requires CUDA toolkit in Docker image (large) | **Chosen** — lowest porting effort, FFT support built in |
| PyTorch | Widely installed, good GPU support | Different API, would need rewrite not port. FFT API differs from scipy. Massive dependency for a CLI tool | Rejected |
| Numba | JIT compilation, no large dependency | No scipy.fftpack equivalent, would need manual FFT. Per-channel pyramid ops would need custom kernels | Rejected |
| PyCUDA | Direct CUDA access, lightweight | Need to write CUDA kernels manually for every operation (pyrDown, FFT, etc.). High effort | Rejected |

### Versioning: VERSION file vs pyproject.toml vs git-tags-only

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| `VERSION` + build-time injection | Scripts self-contained, shell-readable, Docker-friendly | Two files updated by `/release` (automated) | **Chosen** |
| `pyproject.toml` | Python standard | Requires pip install or toml parsing; scripts aren't a package | Rejected — wrong distribution model |
| Git tags only | No version file at all | Docker builds don't have `.git`; need a file anyway | Rejected |

### Test framework: pytest vs unittest

pytest chosen over unittest — simpler assertions, better fixtures, `parametrize` for tolerance tiers, `importorskip` for optional CuPy tests. De facto standard.

### Common module refactor vs duplication

Keeping `evm.py` and `evm_cuda.py` as self-contained scripts rather than extracting `evm_common.py`. The duplication (color matrices, format_duration, arg parsing, validation) is manageable at this scale. A common module would break the "copy one file" simplicity and require Docker changes. Test coverage on the CPU version covers the shared logic by inspection.

## Tradeoffs and Risks

- **No GPU tests in CI**: CUDA codepath regressions won't be caught automatically. Mitigated by: CI lints CUDA code, shared logic tested via CPU tests, developers test GPU locally.
- **Duplicated code across CPU/GPU scripts**: A fix in one could be forgotten in the other. Mitigated by: code review catches this, and the duplication is bounded (shared functions are stable utility code, not evolving logic).
- **`load_video` guard drops frames silently**: If `CAP_PROP_FRAME_COUNT` is too low, we stop at the reported count. For malformed files this means slightly shorter output. Acceptable for edge case with bad metadata.
- **Starting changelog fresh**: No history before v2.1.0. Acceptable — no old versions are tagged or downloadable.

## Open Questions

None — all decisions resolved during grill session.
