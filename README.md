# 法规文档 RAG 后端

这是一个面向法规、制度、政策类文档的本地 RAG 系统。当前仓库不是单一 Python 包，而是一套由 Docker Compose 编排的后端服务栈：主应用负责文档上传、解析、索引、检索和回答生成；Embedding 与 Rerank 拆成独立模型服务；Milvus 存储向量索引；SQLite/FTS 保存本地词法索引和运行状态。

系统当前更偏“法规文档问答与评估平台”，核心目标是让回答严格绑定可追溯证据，尽量避免错引法规、漏引条款或脱离文档生成。

## 当前后端组成

| 模块 | 路径/服务 | 作用 |
| --- | --- | --- |
| RAG 主应用 | `services/rag-app` / `rag-app` | FastAPI 服务，提供 Web 页面、文档管理、检索、问答和评估入口 |
| Embedding 服务 | `services/embedding` / `embedding-service` | 默认加载 `BAAI/bge-m3`，支持 dense embedding，并可返回 BGE-M3 sparse embedding |
| Rerank 服务 | `services/rerank` / `rerank-service` | 默认加载 `BAAI/bge-reranker-base`，对候选片段重排序 |
| 向量库 | `milvus` | 保存文档 chunk 的 dense/sparse 向量与元数据 |
| Milvus 依赖 | `etcd`、`minio` | Milvus standalone 运行依赖 |
| 可选 LLM | `ollama` | 可作为本地 OpenAI-compatible LLM 服务，也可换成外部 OpenAI-compatible 接口 |
| 可选 OCR | 外部 HTTP 服务 | 主应用通过 `OCR_SERVICE_URL` 调用，适用于图片和扫描 PDF |

## 目录结构

```text
rag-architecture/
├─ config/
│  ├─ app.env.example          # 应用配置模板
│  ├─ app.env                  # 本地运行配置，可能包含密钥，不建议提交
│  └─ retrieval_policy.json    # 检索策略配置
├─ database/                   # SQLite/运行数据库目录
├─ docs/                       # 产品、架构、评估和流程文档
├─ documents/                  # 本地资料或样本文档目录
├─ evals/
│  ├─ cases/                   # 评估用例
│  └─ runners/                 # 评估脚本
├─ local-models/               # 本地模型挂载目录
├─ reports/                    # 评估报告输出目录
├─ scripts/                    # 根级辅助脚本
├─ services/
│  ├─ embedding/               # Embedding 微服务
│  ├─ rerank/                  # Rerank 微服务
│  └─ rag-app/                 # RAG 主应用
├─ docker-compose.yml
└─ README.md
```

`services/rag-app/app` 是主应用源码：

```text
app/
├─ api/                        # FastAPI 路由
├─ core/
│  ├─ query/                   # 查询解析、工具路由、召回和问答流程
│  ├─ retrieval/               # 混合检索、过滤、融合、重排、chunk 后处理
│  ├─ source/                  # 文档目标识别、source lock、标题/地区解析
│  └─ evidence/                # 证据选择、证据命中、引用格式化
├─ documents/                  # 文档解析、Document IR、chunking、profile、实体注册
├─ runtime/                    # 运行上下文、依赖装配、启动初始化
├─ services/                   # LLM、OCR、Embedding、Rerank 和文档服务封装
├─ storage/                    # SQLite、Milvus、task store
├─ utils/
├─ config.py
└─ schemas.py
```

## 数据流

### 文档上传与索引

1. 用户通过 `/documents` 或 `/documents/upload` 上传文本或文件。
2. 主应用探测文件类型、大小、页数、解析路线和元数据。
3. 解析器生成 Document IR，尽量保留标题、章节、条款、页码、表格和版面信息。
4. 系统构建 document profile，包括标题别名、地区、主题词、日期、文档类型等。
5. 文档被切分为 chunk，并写入：
   - SQLite/FTS：本地词法索引、文档状态、任务状态。
   - Milvus：dense vector、可选 sparse vector、chunk 元数据。
6. 索引版本发布后，`/query` 和 `/retrieve` 可以召回新文档。

支持的主要文件类型：

```text
.txt .md .markdown .log .json .csv .pdf .doc .docx .xlsx
.png .jpg .jpeg .bmp .tif .tiff .gif
```

图片、扫描 PDF、低文本质量 PDF 会依赖外部 OCR 服务。

### 查询与回答

1. `/query` 接收用户问题。
2. Query Core 执行查询解析、意图/工具路由、问题分解、抽象概念展开等步骤。
3. Source Resolution 尝试锁定目标法规或候选文档，避免跨源误召回。
4. Retrieval 使用 Milvus dense/sparse 召回、SQLite 词法召回、结构化条款信号和融合排序。
5. Rerank 服务在需要时对候选 chunk 或 source 做二次排序。
6. Evidence 模块筛选最终证据，进行证据门控和引用格式化。
7. LLM 仅基于证据上下文生成回答，并返回 `answer`、`sources`、`documents`、`metadata` 等字段。

## 快速启动

### 1. 准备配置

```bash
cp config/app.env.example config/app.env
```

至少确认以下配置：

| 配置 | 说明 |
| --- | --- |
| `LLM_PROVIDER` | 默认按 OpenAI-compatible API 调用 |
| `LLM_API_BASE` | LLM base URL，例如 `http://ollama:11434/v1` 或外部兼容接口 |
| `LLM_CHAT_COMPLETIONS_URL` | 可选，显式指定 chat completions URL 时优先使用 |
| `LLM_MODEL` | 回答生成模型名 |
| `LLM_API_KEY` | 外部接口需要时填写；本地 Ollama 通常可为空 |
| `EMBEDDING_SERVICE_URL` | Docker 内默认 `http://embedding-service:8000` |
| `RERANK_SERVICE_URL` | Docker 内默认 `http://rerank-service:8000` |
| `MILVUS_HOST` / `MILVUS_PORT` | Docker 内默认 `milvus:19530` |
| `OCR_SERVICE_URL` | 可选 OCR HTTP 地址 |
| `OCR_SHARED_CONTAINER_DIR` | OCR 共享目录的容器内路径 |
| `OCR_SHARED_HOST_DIR` | OCR 共享目录的宿主机路径 |
| `RAG_DATABASE_DIR` | Docker 中由 compose 设置为 `/database` |

`config/app.env` 可能包含密钥和内网地址，不建议提交到 Git。

### 2. 准备本地模型

Compose 默认把根目录的 `local-models` 挂载到容器内 `/models`。建议目录如下：

```text
local-models/
└─ BAAI/
   ├─ bge-m3/
   └─ bge-reranker-base/
```

如果本地没有模型，服务可能尝试联网下载，取决于镜像、网络和运行环境。生产或离线环境建议提前放好模型文件。

### 3. 启动服务

```bash
docker compose up -d
```

查看日志：

```bash
docker compose logs -f rag-app
docker compose logs -f embedding-service
docker compose logs -f rerank-service
```

停止服务：

```bash
docker compose down
```

如需清空 Milvus/MinIO/Etcd/Ollama 等 Docker volume，需要额外使用 `docker compose down -v`。这会删除运行数据，操作前请确认。

## 服务地址

| 服务 | 宿主机地址 | 容器内地址/说明 |
| --- | --- | --- |
| RAG Web/API | `http://localhost:8080` | 主应用入口 |
| Embedding | `http://localhost:8001` | 容器内为 `embedding-service:8000` |
| Rerank | `http://localhost:8002` | 容器内为 `rerank-service:8000` |
| Milvus | `localhost:19530` | 向量库 |
| MinIO API | `http://localhost:9000` | Milvus 对象存储 |
| MinIO Console | `http://localhost:9001` | 默认账号密码见 compose |
| Ollama | `http://localhost:11434` | 可选本地 LLM |

健康检查：

```bash
curl http://localhost:8080/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## API

### 问答

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "这份文件里关于处罚的规定是什么？",
    "user_id": "anonymous",
    "top_k": 10,
    "enable_rerank": true
  }'
```

响应主体由 `QueryResponse` 定义：

```json
{
  "answer": "...",
  "sources": [],
  "metadata": {},
  "documents": [],
  "retrieved_contexts": []
}
```

### 只检索不生成

```bash
curl -X POST http://localhost:8080/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "适用范围",
    "user_id": "anonymous",
    "top_k": 10,
    "enable_rerank": true
  }'
```

### 上传纯文本

```bash
curl -X POST http://localhost:8080/documents \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "example.txt",
    "content": "文档正文",
    "metadata": {
      "source": "manual"
    }
  }'
```

### 上传文件

```bash
curl -X POST http://localhost:8080/documents/upload \
  -F "file=@example.pdf"
```

### 文档与任务

```bash
curl http://localhost:8080/documents
curl http://localhost:8080/documents/example.pdf
curl http://localhost:8080/tasks
curl http://localhost:8080/tasks/{task_id}
curl -X POST http://localhost:8080/documents/{task_id}/retry
curl -X DELETE http://localhost:8080/documents/example.pdf
```

## 模型服务 API

Embedding 服务：

```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["法规适用范围"],
    "normalize": true,
    "batch_size": 32,
    "return_sparse": true
  }'
```

Rerank 服务：

```bash
curl -X POST http://localhost:8002/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "处罚规定",
    "documents": ["第一条 ...", "第二条 ..."],
    "top_n": 2,
    "batch_size": 16
  }'
```

## 本地开发

主应用可以在本地 Python 环境中运行，但仍需要可访问的 Milvus、Embedding、Rerank 和 LLM/OCR 服务。

```bash
cd services/rag-app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

如果主应用在宿主机直接运行，常见配置需要改成宿主机端口：

```env
APP_ENV=test_local
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
EMBEDDING_SERVICE_URL=http://127.0.0.1:8001
RERANK_SERVICE_URL=http://127.0.0.1:8002
```

## 索引维护脚本

脚本位于 `services/rag-app/scripts`。

| 脚本 | 作用 | 风险 |
| --- | --- | --- |
| `list_milvus_sources.py` | 列出 Milvus 中的 source、chunk 数量和样本文本 | 只读 |
| `diagnose_milvus_recall.py` | 直接调用 embedding 并搜索 Milvus，诊断一阶段召回 | 只读 |
| `publish_pending_indexes.py` | 发布 pending document index version | 会修改索引状态 |
| `rebuild_all_indexes.py` | 从已有 Document IR 重建 profile、SQLite 和 Milvus 索引 | 高风险，谨慎使用 |

示例：

```bash
cd services/rag-app
python scripts/list_milvus_sources.py --limit 1000 --samples 5 --show-escaped
python scripts/diagnose_milvus_recall.py --query "处罚规定" --top-k 10
```

## 评估

当前评估入口是：

```text
evals/runners/run_documents_eval_pipeline.py
```

默认用例：

```text
evals/cases/documents_leveled_query_dataset.json
```

运行前需要确保 RAG API 已启动并可访问：

```bash
python evals/runners/run_documents_eval_pipeline.py \
  --base-url http://127.0.0.1:8080 \
  --run-name local_smoke \
  --limit 10
```

常用输出位于：

```text
reports/evals/documents_leveled_pipeline/{run_name}/
├─ cases.jsonl
├─ batch_results.json
├─ metrics_report.json
├─ metrics_rows.csv
├─ metrics_rows.xlsx
└─ dashboard.md
```

评估指标覆盖检索、生成和法规引用：

| 类别 | 指标 |
| --- | --- |
| 检索质量 | Context Precision、Context Recall、MRR、NDCG |
| 生成质量 | Faithfulness、Answer Relevance、Unsupported Claim Rate |
| 法规引用 | Citation Recall、Citation Precision、Citation Exact Match |
| 运行成本 | 延迟、token 估算或实际 usage |

可选启用 LLM Judge：

```env
ENABLE_EVAL_LLM_AS_JUDGE=true
EVAL_JUDGE_BACKEND=openai
JUDGE_BASE_URL=...
JUDGE_MODEL=...
JUDGE_API_KEY=...
```

## 测试

主应用测试集中在 `services/rag-app/tests`。

```bash
cd services/rag-app
python -m pytest tests -q -p no:cacheprovider
```

快速语法检查：

```powershell
cd services/rag-app
python -m py_compile (Get-ChildItem app -Recurse -File -Filter *.py | Select-Object -ExpandProperty FullName)
```

## 运行数据与 Git

以下目录或文件属于运行数据、模型文件或生成产物，通常不应提交：

```text
database/
logs/
local-models/
reports/
services/rag-app/data/
*.db
*.sqlite
__pycache__/
.pytest_cache/
```

建议提交：

```text
services/* 源码
config/app.env.example
config/retrieval_policy.json
docs/
evals/cases/
evals/runners/
必要的脚本和测试
```

不建议提交：

```text
config/app.env
模型权重
数据库文件
上传文件缓存
运行日志
临时评估报告
```

## 常见问题

### 主应用启动后 `/health` 正常，但问答失败

优先检查：

1. `LLM_API_BASE` 或 `LLM_CHAT_COMPLETIONS_URL` 是否能从 `rag-app` 容器访问。
2. `LLM_MODEL` 是否是目标服务可用模型。
3. 外部 LLM 是否需要 `LLM_API_KEY`。
4. `docker compose logs -f rag-app` 中是否有 LLM 请求错误、超时或 JSON 解析错误。

### Embedding/Rerank 服务启动慢或失败

优先检查：

1. `local-models/BAAI/bge-m3` 和 `local-models/BAAI/bge-reranker-base` 是否存在。
2. Docker 是否可用 NVIDIA GPU runtime。
3. 镜像是否允许联网下载模型。
4. 宿主机内存和显存是否足够。

### 文档上传后检索不到

按顺序排查：

1. `GET /documents` 查看文档和任务状态。
2. `GET /tasks/{task_id}` 查看解析或索引错误。
3. `python scripts/list_milvus_sources.py` 确认 Milvus 中是否写入 chunk。
4. `python scripts/diagnose_milvus_recall.py --query "..."` 检查一阶段召回。
5. 确认 `database/lexical_index.db` 存在且不是空库。

### PDF 或图片解析质量差

检查：

1. `OCR_SERVICE_URL` 是否配置并可从容器访问。
2. `OCR_SHARED_CONTAINER_DIR` 和 `OCR_SHARED_HOST_DIR` 是否映射到同一共享目录。
3. `PDF_OCR_MAX_TEXT_CHARS_PER_PAGE` 是否适合当前 PDF。
4. 文档是否为扫描件、双栏版式或表格密集型文件。


