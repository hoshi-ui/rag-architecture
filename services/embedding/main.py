from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import time
import json

try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

app = FastAPI(
    title="Embedding Service",
    description="提供文本向量化服务",
    version="1.0.0"
)

# 加载模型
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
MODEL = SentenceTransformer(MODEL_NAME)


class EmbeddingRequest(BaseModel):
    """嵌入请求"""
    texts: List[str]
    normalize: Optional[bool] = True
    batch_size: Optional[int] = 32


class EmbeddingResponse(BaseModel):
    """嵌入响应"""
    embeddings: List[List[float]]
    model: str
    usage: dict


@app.get("/")
async def root():
    return {"service": "Embedding Service", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": True
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """生成文本嵌入"""
    try:
        start_time = time.time()
        
        # 生成嵌入
        embeddings = MODEL.encode(
            request.texts,
            normalize_embeddings=request.normalize,
            batch_size=request.batch_size,
            show_progress_bar=True
        )
        
        # 转换格式
        embeddings_list = embeddings.tolist()
        
        # 返回结果
        return EmbeddingResponse(
            embeddings=embeddings_list,
            model=MODEL_NAME,
            usage={
                "total_tokens": sum(len(t) for t in request.texts),
                "embedding_dimension": len(embeddings[0]),
                "time_seconds": time.time() - start_time
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """列出可用模型"""
    return {
        "current": MODEL_NAME,
        "supported": [
            "BAAI/bge-m3",
            "BAAI/bge-base-zh",
            "sentence-transformers/all-MiniLM-L6-v2"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
