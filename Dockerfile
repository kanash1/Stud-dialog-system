# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

# APT с кэшем
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential libgl1 && \
    rm -rf /var/lib/apt/lists/*

# pip с кэшем
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY wait-for-it.sh .
COPY app/ ./app/
RUN chmod +x wait-for-it.sh

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]