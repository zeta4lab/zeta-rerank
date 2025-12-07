FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml ./
COPY server.py ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Environment variables
ENV RERANK_MODEL="Qwen/Qwen3-Reranker-0.6B"
ENV RERANK_DEVICE="cpu"
ENV RERANK_PORT="9046"
ENV PYTORCH_ENABLE_MPS_FALLBACK="1"

EXPOSE 9046

CMD ["python", "server.py"]
