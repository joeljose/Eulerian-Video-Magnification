FROM python:3.11.15-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ARG UID
ARG GID
ARG UNAME

RUN groupadd -g ${GID} ${UNAME} && \
    useradd -m -u ${UID} -g ${GID} ${UNAME}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evm.py .

USER ${UNAME}

ENTRYPOINT ["python", "-u", "evm.py"]
