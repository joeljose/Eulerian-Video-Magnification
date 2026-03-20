# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
