FROM python:3.11.16-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

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
