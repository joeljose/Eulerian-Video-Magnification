FROM python:3.11.18-slim@sha256:ffe4b0c76963e64f05484d74dbb3f67fad1effcf3c5c22c2235f0fafc8a9c98f

# Fixed non-root user; for bind mounts run with --user "$(id -u):$(id -g)"
RUN useradd -m -u 1000 app

WORKDIR /app

# Hash-locked dependencies; regenerate requirements.lock as described in
# CONTRIBUTING.md when requirements*.txt change
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY evm.py evm_cuda.py pyproject.toml VERSION ./
COPY tests/ tests/
COPY scripts/ scripts/

RUN chown -R app:app /app

ARG VERSION
LABEL version=${VERSION}

USER app

ENTRYPOINT ["python", "-u", "evm.py"]
