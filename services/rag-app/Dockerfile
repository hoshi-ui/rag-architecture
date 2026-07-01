FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8080
ENV WAIT_FOR_MILVUS=true
ENV MILVUS_CONNECT_TIMEOUT_SEC=180
ENV MILVUS_CONNECT_RETRY_INTERVAL_SEC=2

RUN apt-get update && apt-get install -y \
    antiword \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 先复制 web 目录到 /app/web
COPY web /app/web

# 再复制后端代码到 /app
COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f "http://localhost:${APP_PORT}/health" || exit 1

CMD ["sh", "-c", "if [ \"${WAIT_FOR_MILVUS}\" = \"true\" ]; then python wait_for_service.py; fi && uvicorn main:app --host ${APP_HOST} --port ${APP_PORT}"]
