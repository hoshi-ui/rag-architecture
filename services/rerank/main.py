from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import time
import logging

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rerank-service")

app = FastAPI(
    title="Rerank Service",
    description="提供文档重排序服务",
    version="1.0.0"
)

# 加载重排序模型
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-reranker-base")
RERANKER = CrossEncoder(MODEL_NAME)


class RerankRequest(BaseModel):
    """重排序请求"""
    query: str
    documents: List[str]
    top_n: Optional[int] = 10
    batch_size: Optional[int] = 16


class ScoredDocument(BaseModel):
    """带分数的文档"""
    content: str
    score: float
    index: int


class RerankResponse(BaseModel):
    """重排序响应"""
    results: List[ScoredDocument]
    model: str
    usage: dict


@app.get("/")
async def root():
    return {"service": "Rerank Service", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """对文档进行重排序"""
    try:
        start_time = time.time()
        
        # 准备输入对
        pairs = [[request.query, doc] for doc in request.documents]
        
        # 计算相关性分数
        scores = RERANKER.predict(pairs, batch_size=request.batch_size, show_progress_bar=False)
        
        # 创建结果
        scored_docs = []
        for i, (doc, score) in enumerate(zip(request.documents, scores)):
            scored_docs.append(ScoredDocument(
                content=doc,
                score=float(score),
                index=i
            ))
        
        # 按分数排序，取前 top_n
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        results = scored_docs[:request.top_n]
        
        logger.info(f"rerank_ok docs={len(request.documents)} top_n={request.top_n} bs={request.batch_size} time={time.time() - start_time:.3f}s")
        return RerankResponse(
            results=results,
            model=MODEL_NAME,
            usage={
                "total_documents": len(request.documents),
                "returned_documents": len(results),
                "time_seconds": time.time() - start_time
            }
        )
    except Exception as e:
        logger.exception(f"rerank_error docs={len(request.documents)} top_n={request.top_n} bs={request.batch_size}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
