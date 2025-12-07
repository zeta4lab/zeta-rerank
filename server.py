# -*- coding: utf-8 -*-
"""BGE Reranker Server - CrossEncoder 기반

vLLM Score API 호환 엔드포인트 제공:
- POST /v1/score - 단일 쿼리-문서 점수
- POST /v1/rerank - 배치 재순위화

CrossEncoder 방식:
- Instruction 불필요
- 쿼리-문서 쌍을 직접 점수화
- 다국어 지원 (bge-reranker-v2-m3)

Apple Silicon (MPS) 및 CPU 지원
"""
import os
import logging
from typing import List, Optional, Union
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# MPS fallback for Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("RERANK_DEVICE", "auto")  # auto, cpu, mps, cuda
MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
PORT = int(os.getenv("RERANK_PORT", "9046"))

# Global model instance
model: Optional[CrossEncoder] = None


# =============================================================================
# Request/Response Models
# =============================================================================
class ScoreRequest(BaseModel):
    """vLLM Score API 호환 요청"""
    model: str = Field(default="bge-reranker")
    text_1: str = Field(..., description="Query text")
    text_2: str = Field(..., description="Document text")
    instruction: Optional[str] = None  # 무시됨 (호환성 유지)


class ScoreResponse(BaseModel):
    """vLLM Score API 호환 응답"""
    model: str
    score: float
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})


class RerankDocument(BaseModel):
    """재순위화할 문서"""
    id: Optional[str] = None
    text: str


class RerankRequest(BaseModel):
    """배치 재순위화 요청"""
    model: str = Field(default="bge-reranker")
    query: str
    documents: List[Union[str, RerankDocument]]
    instruction: Optional[str] = None  # 무시됨 (호환성 유지)
    top_k: Optional[int] = None
    return_documents: bool = True


class RerankResult(BaseModel):
    """재순위화 결과"""
    index: int
    score: float
    document: Optional[RerankDocument] = None


class RerankResponse(BaseModel):
    """배치 재순위화 응답"""
    model: str
    results: List[RerankResult]
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})


class ModelInfo(BaseModel):
    """모델 정보"""
    id: str
    object: str = "model"
    owned_by: str = "zeta-rerank"


class ModelsResponse(BaseModel):
    """모델 목록 응답"""
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# Model Loading & Inference
# =============================================================================
def get_device() -> str:
    """최적 디바이스 자동 선택"""
    if DEVICE != "auto":
        return DEVICE

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model():
    """BGE Reranker 모델 로드"""
    global model

    device = get_device()
    logger.info(f"Loading model: {MODEL_NAME} on device: {device}")

    try:
        # CrossEncoder 로드
        model = CrossEncoder(
            MODEL_NAME,
            max_length=MAX_LENGTH,
            device=device,
            trust_remote_code=True,
        )

        logger.info(f"Model loaded successfully on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def compute_scores(pairs: List[tuple]) -> List[float]:
    """배치 점수 계산

    Args:
        pairs: [(query, document), ...] 리스트

    Returns:
        [score, ...] 관련성 점수 (sigmoid 적용된 0-1 범위)
    """
    if not pairs:
        return []

    # CrossEncoder.predict()는 배치 처리 지원
    scores = model.predict(pairs, convert_to_numpy=True)

    # numpy array를 list로 변환, sigmoid 적용하여 0-1 범위로
    result = []
    for score in scores:
        # BGE reranker는 logit 값 반환, sigmoid로 확률 변환
        prob = 1 / (1 + pow(2.718281828, -float(score)))
        result.append(prob)

    return result


# =============================================================================
# FastAPI App
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 모델 로드/정리"""
    load_model()
    yield
    # Cleanup (GC handles it)


app = FastAPI(
    title="Zeta Rerank Server",
    description="BGE Reranker using CrossEncoder - vLLM Score API Compatible",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health")
async def health():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": get_device(),
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """사용 가능한 모델 목록"""
    return ModelsResponse(
        data=[
            ModelInfo(id="bge-reranker"),
            ModelInfo(id=MODEL_NAME),
        ]
    )


@app.post("/v1/score", response_model=ScoreResponse)
async def compute_score_endpoint(request: ScoreRequest):
    """단일 쿼리-문서 점수 계산 (vLLM 호환)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        scores = compute_scores([(request.text_1, request.text_2)])

        return ScoreResponse(
            model=request.model,
            score=scores[0],
        )
    except Exception as e:
        logger.error(f"Score computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """배치 문서 재순위화"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 문서 텍스트 추출
        documents = []
        for doc in request.documents:
            if isinstance(doc, str):
                documents.append(RerankDocument(text=doc))
            else:
                documents.append(doc)

        # (query, document) 쌍 생성
        pairs = [(request.query, doc.text) for doc in documents]

        # 배치 점수 계산
        scores = compute_scores(pairs)

        # 결과 생성
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                index=i,
                score=score,
                document=doc if request.return_documents else None,
            ))

        # 점수 기준 정렬
        results.sort(key=lambda x: x.score, reverse=True)

        # top_k 적용
        if request.top_k:
            results = results[:request.top_k]

        return RerankResponse(
            model=request.model,
            results=results,
        )
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main
# =============================================================================
def main():
    """서버 실행"""
    import uvicorn

    logger.info(f"Starting Zeta Rerank Server on port {PORT}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {get_device()}")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
