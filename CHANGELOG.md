# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Unit tests for core CPU functions (color conversion, bandpass filter, pyramid ops)
- `VERSION` file as single source of truth for versioning
- Docker image version labels
- `CHANGELOG.md`
- `requirements-dev.txt` for dev dependencies (pytest, ruff)
- Design doc (`docs/design/evm-hardening.md`)
- pytest step in CI workflow

### Fixed
- `load_video` buffer overflow when `CAP_PROP_FRAME_COUNT` underreports
- `docker-build-cuda.sh` fragile version extraction replaced with simple grep

### Changed
- Build scripts read version from `VERSION` file instead of grepping Python source
- `cupy-cuda12x` dependency pinned with upper bound (`<14`)
