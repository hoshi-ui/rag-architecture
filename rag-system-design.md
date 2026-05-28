# RAG 系统架构设计

## 🎯 设计目标

构建一个面向**国内 ToB 场景**的检索增强生成（RAG）系统，支持：
- 企业文档（PDF、Word、Excel、PPT）
- 钉钉/飞书等多渠道消息
- 实时知识库更新
- 访问控制和权限管理
- 中文语义理解优化

---

## 🏗️ 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户层 (User Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│  钉钉  │  飞书  │  企业微信  │  Web 界面  │  API 接口               │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      网关层 (Gateway Layer)                      │
├─────────────────────────────────────────────────────────────────┤
│  OpenClaw Gateway  │  WebSocket  │  HTTP  │  Authentication      │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    应用服务层 (Application Layer)                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  查询路由     │  │  权限校验     │  │  上下文管理   │          │
│  │  Router      │  │  Permission   │  │  Context      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    检索层 (Retrieval Layer)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Query       │  │  Hybrid      │  │  Rerank      │          │
│  │  Processor   │  │  Search      │  │  Model       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ↓                    ↓                    ↓                    │
│  文档级中文 fallback: 标题/别名/文件名子串 + 文档级词项重叠 +     │
│  document-title 合成召回，为主检索链提供 doc-level prior        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Vector Database                         │       │
│  │         (Milvus / Qdrant / Weaviate)                 │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                  知识库层 (Knowledge Base Layer)                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  文档库       │  │  结构化数据   │  │  消息历史     │          │
│  │  Documents   │  │  Structured  │  │  Messages     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 技术栈选型

### 1. 核心组件

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| **RAG 框架** | OpenClaw | 多通道集成，会话管理 |
| **向量数据库** | Milvus (自托管) | 支持中文分词，高性能 |
| **嵌入模型** | bge-m3 / text2vec | 中文优化，多语言支持 |
| **LLM** | Qwen2.5 / ChatGLM3 | 中文理解优秀，可本地部署 |
| **重排序** | bge-reranker | 提高检索准确率 |
| **Web 服务器** | FastAPI | 异步处理，高性能 |

### 2. 基础设施

```yaml
# docker-compose.yml
services:
  # 向量数据库
  milvus:
    image: milvusdb/milvus:latest
    volumes:
      - milvus_data:/var/lib/milvus

  # 嵌入服务
  embedding-service:
    build: ./services/embedding
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # 重排序服务
  rerank-service:
    build: ./services/rerank
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # 应用服务
  rag-app:
    build: ./services/rag-app
    environment:
      - REDIS_URL=redis://redis:6379
      - MILVUS_HOST=milvus
    depends_on:
      - milvus
      - redis

  # 缓存
  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data
```

---

## 🔄 数据流程

### 1. 文档摄入管道 (Ingestion Pipeline)

```python
# ingestion.py
class DocumentIngestionPipeline:
    """文档摄入管道"""
    
    def process(self, file_path: str):
        """处理文档"""
        
        # 步骤 1: 解析文档
        document = self.parser.parse(file_path)
        
        # 步骤 2: 分块 (10-20% 重叠)
        chunks = self.chunker.chunk(
            document.text,
            chunk_size=500,
            overlap=100
        )
        
        # 步骤 3: 添加元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                "source": file_path,
                "chunk_id": i,
                "created_at": datetime.now(),
                "department": self.extract_department(file_path),
                "access_level": self.extract_access_level(file_path)
            }
        
        # 步骤 4: 生成嵌入
        embeddings = self.embedder.encode(chunks)
        
        # 步骤 5: 存储到向量数据库
        self.vector_db.upsert(
            ids=[f"{file_path}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks]
        )
        
        return len(chunks)
```

### 2. 查询处理管道 (Query Pipeline)

```python
# query.py
class QueryProcessingPipeline:
    """查询处理管道"""
    
    def __init__(self, config):
        self.query_expander = QueryExpander()
        self.search_engine = HybridSearchEngine()
        self.reranker = Reranker()
        self.context_generator = ContextGenerator()
    
    def process(self, query: str, user_id: str, context: dict):
        """处理查询"""
        
        # 步骤 1: 查询理解
        understood_query = self.query_expander.expand(
            query,
            user_context=context.get("user_profile")
        )
        
        # 步骤 2: 权限过滤检索
        filtered_docs = self.search_engine.hybrid_search(
            query=understood_query,
            filters={
                "access_level": self.get_user_access(user_id),
                "department": context.get("department")
            },
            top_k=50
        )
        
        # 步骤 3: 重排序
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=filtered_docs,
            top_k=10
        )
        
        # 步骤 4: 生成上下文
        context = self.context_generator.format(
            reranked_docs,
            max_tokens=2000
        )
        
        # 步骤 5: 生成回答
        answer = self.llm.generate(
            prompt=self.build_prompt(query, context),
            temperature=0.3
        )
        
        return {
            "answer": answer,
            "sources": [doc.source for doc in reranked_docs],
            "similarity_scores": [doc.score for doc in reranked_docs]
        }
```

---

## 🔐 安全与权限

### 1. 访问控制策略

```python
# access_control.py
class AccessControl:
    """访问控制"""
    
    def filter_by_access(self, user_id: str):
        """基于用户权限过滤"""
        
        user_permissions = self.get_user_permissions(user_id)
        
        return {
            "access_level": user_permissions.get("level", "public"),
            "allowed_departments": user_permissions.get("departments", []),
            "restricted_sources": user_permissions.get("blocked", [])
        }
    
    def apply_filters(self, query: dict, user_filters: dict):
        """应用过滤器到查询"""
        
        filters = query.copy()
        filters["metadata.access_level"] = {
            "$lte": self.get_level_value(user_filters["access_level"])
        }
        
        if user_filters["allowed_departments"]:
            filters["metadata.department"] = {
                "$in": user_filters["allowed_departments"]
            }
        
        if user_filters["restricted_sources"]:
            filters["metadata.source"] = {
                "$nin": user_filters["restricted_sources"]
            }
        
        return filters
```

### 2. PII 检测

```python
# pii_detector.py
class PIIDetector:
    """PII 检测器"""
    
    def scan(self, text: str) -> List[PII]:
        """扫描敏感信息"""
        
        patterns = {
            "phone": r"1[3-9]\d{9}",
            "id_card": r"\d{17}[\dX]",
            "email": r"\w+@\w+\.\w+",
            "bank_card": r"\d{16,19}"
        }
        
        pii_list = []
        for ptype, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                pii_list.append(PII(
                    type=ptype,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))
        
        return pii_list
```

---

## 📊 评估与监控

### 1. 评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **Recall@K** | 召回率 | > 0.8 |
| **Precision@K** | 精确率 | > 0.7 |
| **MRR** | 平均倒数排名 | > 0.6 |
| **Response Time** | 响应时间 | < 2s |
| **Hallucination Rate** | 幻觉率 | < 5% |

### 2. 监控配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rag-monitor'
    static_configs:
      - targets: ['rag-app:8080']
    metrics_path: '/metrics'
    
  - job_name: 'milvus'
    static_configs:
      - targets: ['milvus:9091']
```

---

## 🚀 部署步骤

### 1. 本地开发环境

```bash
# 克隆项目
git clone https://github.com/your-org/rag-system
cd rag-system

# 安装依赖
pip install -r requirements.txt

# 启动服务
docker-compose up -d

# 初始化向量数据库
python scripts/init_vector_db.py

# 启动应用
python main.py
```

### 2. 生产环境部署

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置数据库连接等

# 2. 构建镜像
docker-compose build

# 3. 部署到服务器
docker-compose -f docker-compose.prod.yml up -d

# 4. 健康检查
curl http://localhost:8080/health
```

---

## 📝 最佳实践

1. **分块策略** - 根据文档类型调整分块大小，技术文档用小块，报告用大块
2. **混合搜索** - 以稠密检索为主，叠加文档级中文 fallback：标题/别名/文件名子串匹配、文档级词项重叠召回、document-title 合成召回，并将结果作为 doc-level prior 注入主检索链
3. **定期评估** - 每周更新评估数据集，监控性能变化
4. **增量更新** - 支持文档增量更新，避免全量重新索引
5. **缓存机制** - 缓存热门查询的响应，提高响应速度
6. **A/B 测试** - 测试不同嵌入模型和搜索参数的效果
7. **中文 fallback 边界** - source-aware chunk fuzzy 可作为实验开关保留，但默认不启用；正式兜底能力保持在 doc-level，避免增加时延却没有稳定收益

---

## 🔧 扩展建议

### Phase 1 (基础功能)
- ✅ 文档解析和分块
- ✅ 向量检索
- ✅ 简单问答

### Phase 2 (优化功能)
- 混合搜索
- 重排序
- 权限控制

### Phase 3 (高级功能)
- 多轮对话
- 知识图谱
- 自动评估

---

**架构版本：** v1.0  
**最后更新：** 2026-04-14  
**维护者：** OpenClaw Team
