import os
import socket
import sys
import time


def _float_env(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def main() -> int:
    host = os.getenv("WAIT_HOST") or os.getenv("MILVUS_HOST", "milvus")
    port_raw = os.getenv("WAIT_PORT") or os.getenv("MILVUS_PORT", "19530")
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        print(f"wait_for_service_invalid_port: {port_raw}", file=sys.stderr, flush=True)
        return 2

    timeout = _float_env("WAIT_TIMEOUT_SEC", os.getenv("MILVUS_CONNECT_TIMEOUT_SEC", "180"))
    interval = max(0.2, _float_env("WAIT_RETRY_INTERVAL_SEC", os.getenv("MILVUS_CONNECT_RETRY_INTERVAL_SEC", "2")))
    deadline = time.monotonic() + max(0.0, timeout)

    while True:
        try:
            with socket.create_connection((host, port), timeout=min(interval, 5.0)):
                print(f"wait_for_service_ok: {host}:{port}", flush=True)
                return 0
        except OSError as exc:
            if time.monotonic() >= deadline:
                print(f"wait_for_service_timeout: {host}:{port} err={exc}", file=sys.stderr, flush=True)
                return 1
            print(f"wait_for_service_retry: {host}:{port} err={exc}", flush=True)
            time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
