# -*- coding: utf-8 -*-
"""Qwen3 Reranker Server - Transformers 기반

vLLM Score API 호환 엔드포인트 제공:
- POST /v1/score - 단일 쿼리-문서 점수
- POST /v1/rerank - 배치 재순위화

Qwen3-Reranker의 공식 구현 방식 사용:
- "yes"/"no" 토큰의 logits로 점수 계산
- log_softmax → "yes" 확률 = 관련성 점수

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
from transformers import AutoModelForCausalLM, AutoTokenizer

# MPS fallback for Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
DEVICE = os.getenv("RERANK_DEVICE", "auto")  # auto, cpu, mps, cuda
MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
PORT = int(os.getenv("RERANK_PORT", "9046"))
DEFAULT_INSTRUCTION = os.getenv(
    "RERANK_INSTRUCTION",
    "OMP 명령어 검색 쿼리가 주어지면, 가장 관련 있는 명령어를 찾아주세요"
)

# Global model instances
model = None
tokenizer = None
token_true_id = None
token_false_id = None


# =============================================================================
# Request/Response Models
# =============================================================================
class ScoreRequest(BaseModel):
    """vLLM Score API 호환 요청"""
    model: str = Field(default="qwen3-reranker")
    text_1: str = Field(..., description="Query text")
    text_2: str = Field(..., description="Document text")
    instruction: Optional[str] = None


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
    model: str = Field(default="qwen3-reranker")
    query: str
    documents: List[Union[str, RerankDocument]]
    instruction: Optional[str] = None
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
    """Qwen3-Reranker 모델 로드"""
    global model, tokenizer, token_true_id, token_false_id

    device = get_device()
    logger.info(f"Loading model: {MODEL_NAME} on device: {device}")

    try:
        # Tokenizer 로드 (padding_side='left' 필수)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side='left',
            trust_remote_code=True,
        )

        # 모델 로드
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,  # MPS는 float16 제한 있음
                trust_remote_code=True,
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
            )

        model.eval()

        # "yes"/"no" 토큰 ID 저장
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        token_false_id = tokenizer.convert_tokens_to_ids("no")

        logger.info(f"Model loaded successfully")
        logger.info(f"Token IDs - yes: {token_true_id}, no: {token_false_id}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def format_instruction(query: str, document: str, instruction: Optional[str] = None) -> str:
    """Qwen3 Reranker 입력 포맷"""
    inst = instruction or DEFAULT_INSTRUCTION
    return f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {document}"


@torch.no_grad()
def compute_scores(pairs: List[tuple], instruction: Optional[str] = None) -> List[float]:
    """배치 점수 계산

    Args:
        pairs: [(query, document), ...] 리스트
        instruction: 커스텀 instruction (optional)

    Returns:
        [score, ...] 0-1 범위의 관련성 점수
    """
    if not pairs:
        return []

    scores = []
    device = next(model.parameters()).device

    # 개별 처리 (배치 패딩 문제 회피)
    for query, document in pairs:
        text = format_instruction(query, document, instruction)

        # 토크나이징
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 모델 추론
        outputs = model(**inputs)

        # 마지막 토큰의 logits에서 yes/no 점수 추출
        logits = outputs.logits[0, -1, :]
        true_logit = logits[token_true_id]
        false_logit = logits[token_false_id]

        # softmax로 확률 계산
        stacked = torch.stack([false_logit, true_logit])
        probs = torch.nn.functional.softmax(stacked, dim=0)

        # "yes" 확률 = 관련성 점수
        score = probs[1].cpu().item()

        # NaN 체크
        if score != score:  # NaN check
            score = 0.0

        scores.append(score)

    return scores


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
    description="Qwen3 Reranker using Transformers - vLLM Score API Compatible",
    version="0.1.0",
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
            ModelInfo(id="qwen3-reranker"),
            ModelInfo(id=MODEL_NAME),
        ]
    )


@app.post("/v1/score", response_model=ScoreResponse)
async def compute_score_endpoint(request: ScoreRequest):
    """단일 쿼리-문서 점수 계산 (vLLM 호환)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        scores = compute_scores(
            [(request.text_1, request.text_2)],
            instruction=request.instruction,
        )

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
        scores = compute_scores(pairs, instruction=request.instruction)

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
