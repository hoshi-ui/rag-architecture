# RAG 系统架构

基于 OpenClaw 的企业级检索增强生成（RAG）系统。

## 当前阶段文档

- [STAGE-MEMORY-SUMMARY.md](STAGE-MEMORY-SUMMARY.md)
- [STAGE-ACCEPTANCE.md](STAGE-ACCEPTANCE.md)
- [DEFAULT-RUN-STRATEGY.md](DEFAULT-RUN-STRATEGY.md)

## 📁 项目结构

```
rag-architecture/
├── docker-compose.yml          # Docker 编排文件
├── config/
│   ├── app.env                 # 应用环境变量
│   ├── retrieval_policy.json   # 检索策略层配置（route/qtype/profile/filter）
│   └── prometheus.yml          # Prometheus 监控配置
├── services/
│   ├── embedding/              # 嵌入服务
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   ├── rerank/                 # 重排序服务
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   └── rag-app/                # 主应用
│       ├── Dockerfile
│       ├── requirements.txt
│       └── main.py
├── database/
│   └── schema.sql              # 数据库 Schema
├── tests/
│   └── test_rag.py             # 测试套件
└── README.md
```

## 🚀 快速开始

### 1. 环境要求

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU (可选，用于加速)

### 2. 启动服务

```bash
# 克隆项目
git clone <repo-url>
cd rag-architecture

# 配置环境变量
cp config/app.env.example config/app.env
# 编辑 app.env 文件

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 3. 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| RAG 应用 | http://localhost:8080 | 主 API |
| Milvus | http://localhost:19530 | 向量数据库 |
| MinIO | http://localhost:9001 | 对象存储 |
| Prometheus | http://localhost:9090 | 监控数据 |
| Grafana | http://localhost:3000 | 监控仪表板 |

## 🧪 测试

```bash
# 运行单元测试
pytest tests/ -v

# 运行集成测试
pytest tests/test_rag.py -v --tb=short

# 运行性能测试
pytest tests/test_performance.py -v
```

## 📋 API 文档

### 健康检查

```bash
curl http://localhost:8080/health
```

### 查询文档

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是 RAG 系统？",
    "user_id": "test_user",
    "top_k": 5
  }'
```

### 上传文档

```bash
curl -X POST http://localhost:8080/documents \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "document.pdf",
    "content": "文档内容...",
    "metadata": {
      "department": "engineering"
    }
  }'
```

## 🔧 配置说明

### 核心参数

- `CHUNK_SIZE`: 文档分块大小（默认 500 字符）
- `OVERLAP`: 重叠部分（默认 100 字符）
- `TOP_K`: 初始检索数量（默认 50）
- `RERANK_TOP_K`: 重排序后数量（默认 10）

### 检索策略层

高影响的检索前决策已开始从代码分支抽离到声明式策略文件：

- `config/retrieval_policy.json`: 控制 query route、question type、rerank profile 触发、query filter、weak-reference 识别
- 优先修改该文件来调整规则源，而不是直接改 `services/rag-app/main.py` 中的关键词表
- 如需覆盖默认位置，可通过环境变量 `RETRIEVAL_POLICY_FILE` 指定策略文件路径

### 模型配置

- `embedding_model`: BAAI/bge-m3 (中文优化)
- `rerank_model`: BAAI/bge-reranker-base
- `llm_model`: Qwen2.5-7B-Instruct

## 📊 监控

### Prometheus 指标

- `rag_query_duration_seconds`: 查询耗时
- `rag_retrieval_count`: 检索文档数
- `rag_embedding_latency`: 嵌入生成耗时

### Grafana 仪表板

使用 `config/grafana-dashboards.json` 导入仪表板。

## 🔒 安全

- 启用访问控制
- PII 检测
- 查询过滤
- 权限验证

## 📈 性能优化

- 使用 GPU 加速嵌入和重排序
- 启用缓存机制
- 优化分块策略
- 使用混合搜索

## 🐛 故障排除

### 常见问题

1. **服务无法启动**
   - 检查端口是否被占用
   - 查看日志：`docker-compose logs -f`

2. **Milvus 连接失败**
   - 检查网络配置
   - 验证凭据是否正确

3. **嵌入生成慢**
   - 检查 GPU 是否可用
   - 调整 batch_size

### 日志位置

```bash
# 应用日志
docker-compose logs -f rag-app

# 向量数据库日志
docker-compose logs -f milvus
```

## 📚 参考资料

- [OpenClaw 文档](https://docs.openclaw.ai)
- [Milvus 文档](https://milvus.io/docs)
- [BGE 模型文档](https://github.com/FlagOpen/FlagEmbedding)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

**版本**: 1.0.0  
**最后更新**: 2026-03-18
