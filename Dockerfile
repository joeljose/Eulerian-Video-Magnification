FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ARG UID
ARG GID
ARG UNAME

RUN groupadd -g ${GID} ${UNAME} && \
    useradd -m -u ${UID} -g ${GID} ${UNAME}

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY evm.py .
COPY tests/ tests/

RUN chown -R ${UID}:${GID} /app

ARG VERSION
LABEL version=${VERSION}

USER ${UNAME}

ENTRYPOINT ["python", "-u", "evm.py"]
