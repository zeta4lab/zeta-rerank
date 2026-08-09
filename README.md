# Zeta Rerank Server

Qwen3 Reranker Server using Sentence Transformers - vLLM Score API Compatible

## Features

- vLLM Score API 호환 (`/v1/score`)
- 배치 재순위화 (`/v1/rerank`)
- Apple Silicon (MPS) 지원
- CPU/CUDA 지원

## Quick Start

```bash
# Install
poetry install

# Run server
poetry run python server.py

# Or with environment variables
RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B \
RERANK_PORT=9046 \
poetry run python server.py
```

## API Endpoints

### POST /v1/score

```json
{
  "model": "qwen3-reranker",
  "text_1": "검색 쿼리",
  "text_2": "문서 내용"
}
```

Response:
```json
{
  "model": "qwen3-reranker",
  "score": 0.85
}
```

### POST /v1/rerank

```json
{
  "model": "qwen3-reranker",
  "query": "검색 쿼리",
  "documents": ["문서1", "문서2", "문서3"],
  "top_k": 3
}
```

## Docker

```bash
docker-compose up -d
```

## Environment Variables

- `RERANK_MODEL`: HuggingFace 모델 ID (default: `Qwen/Qwen3-Reranker-0.6B`)
- `RERANK_DEVICE`: `auto`, `cpu`, `mps`, `cuda` (default: `auto`)
- `RERANK_PORT`: 서버 포트 (default: `9046`)
- `RERANK_MAX_LENGTH`: 최대 시퀀스 길이 (default: `512`)

## 라이선스

[Apache License 2.0](LICENSE)

Copyright 2026 **제타포랩(zeta4lab)**

- 대표: 최강유
- https://zeta4.net

자세한 저작권 및 고지 사항은 [NOTICE](NOTICE)를 참고하세요.
