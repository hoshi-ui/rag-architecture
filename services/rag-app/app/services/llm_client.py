import asyncio
import json
from typing import Any, Dict, List, Optional

import requests


class LlmClient:
    def __init__(self, config: Any) -> None:
        self.config = config

    def available(self) -> bool:
        return bool(self.config.LLM_CHAT_COMPLETIONS_URL or self.config.LLM_API_BASE)

    def chat_url_candidates(self) -> List[str]:
        if self.config.LLM_CHAT_COMPLETIONS_URL:
            return [self.config.LLM_CHAT_COMPLETIONS_URL]
        base = (self.config.LLM_API_BASE or "").rstrip("/")
        candidates: List[str] = []
        if base:
            candidates.append(f"{base}/chat/completions")
            if not base.endswith("/v1"):
                candidates.append(f"{base}/v1/chat/completions")
            if base.endswith("/v1"):
                candidates.append(f"{base[:-3].rstrip('/')}/chat/completions")
        return candidates

    def extra_body(self) -> Dict[str, Any]:
        raw = (self.config.LLM_EXTRA_BODY or "").strip()
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.config.LLM_API_KEY}"
        return headers

    @staticmethod
    def estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
        chars = 0
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else ""
            if isinstance(content, str):
                chars += len(content)
            else:
                chars += len(json.dumps(content, ensure_ascii=False))
            chars += 8
        # Conservative for Chinese legal text: many chars can be close to one token.
        return max(1, chars)

    def clamp_max_tokens(self, messages: List[Dict[str, Any]], requested: int) -> int:
        context_window = int(getattr(self.config, "LLM_CONTEXT_WINDOW", 0) or 0)
        if context_window <= 0:
            return int(requested)
        safety_margin = max(0, int(getattr(self.config, "LLM_OUTPUT_TOKEN_SAFETY_MARGIN", 2048) or 0))
        input_tokens = self.estimate_message_tokens(messages)
        available = context_window - input_tokens - safety_margin
        return max(1, min(int(requested), available))

    @staticmethod
    def extract_choice_text(choice: Any) -> str:
        if not isinstance(choice, dict):
            return ""

        def collapse_text_blocks(value: Any) -> str:
            if isinstance(value, str):
                return value.strip()
            if not isinstance(value, list):
                return ""
            parts: List[str] = []
            for block in value:
                if isinstance(block, str) and block.strip():
                    parts.append(block.strip())
                    continue
                if not isinstance(block, dict):
                    continue
                text_value = block.get("text") or block.get("content") or block.get("value")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
            return "\n".join(parts).strip()

        msg = choice.get("message") or {}
        if isinstance(msg, dict):
            for value in (msg.get("content"), msg.get("reasoning"), choice.get("text")):
                content = collapse_text_blocks(value)
                if content:
                    return content
        return collapse_text_blocks(choice.get("text"))

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        requested_max_tokens = int(max_tokens or self.config.LLM_MAX_TOKENS)
        output_cap = int(getattr(self.config, "LLM_MAX_OUTPUT_TOKENS_CAP", 0) or 0)
        if output_cap > 0:
            requested_max_tokens = min(requested_max_tokens, output_cap)
        payload: Dict[str, Any] = {
            "model": self.config.LLM_MODEL,
            "messages": messages,
            "temperature": self.config.LLM_TEMPERATURE if temperature is None else temperature,
            "top_p": self.config.LLM_TOP_P if top_p is None else top_p,
            "max_tokens": self.clamp_max_tokens(messages, requested_max_tokens),
            "presence_penalty": self.config.LLM_PRESENCE_PENALTY if presence_penalty is None else presence_penalty,
        }
        for key, value in self.extra_body().items():
            payload.setdefault(key, value)
        for key, value in (extra or {}).items():
            payload[key] = value
        return payload

    def chat_text_sync(self, payload: Dict[str, Any], *, timeout: Optional[int] = None) -> str:
        data = self.chat_response_sync(payload, timeout=timeout)
        if not data:
            return ""
        choices = data.get("choices") or []
        if choices and isinstance(choices, list):
            content = self.extract_choice_text(choices[0])
            if content:
                return content
        raise RuntimeError("LLM chat failed")

    def chat_response_sync(self, payload: Dict[str, Any], *, timeout: Optional[int] = None) -> Dict[str, Any]:
        if not self.available():
            return {}
        last_exc: Optional[BaseException] = None
        endpoint_errors: List[str] = []
        for url in self.chat_url_candidates():
            try:
                response = requests.post(
                    url,
                    headers=self.headers(),
                    json=payload,
                    timeout=int(timeout or self.config.LLM_TIMEOUT),
                )
                if response.status_code == 404:
                    body = response.text[:300].replace("\n", " ")
                    endpoint_errors.append(f"{url} -> 404 {body}")
                    last_exc = RuntimeError("LLM endpoint 404: " + " | ".join(endpoint_errors))
                    continue
                if response.status_code >= 400:
                    body = response.text[:500].replace("\n", " ")
                    raise RuntimeError(f"LLM request failed: {response.status_code} {url} {body}")
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("LLM chat failed")

    async def chat_text(self, payload: Dict[str, Any], *, timeout: Optional[int] = None) -> str:
        return await asyncio.to_thread(self.chat_text_sync, payload, timeout=timeout)

    async def chat_response(self, payload: Dict[str, Any], *, timeout: Optional[int] = None) -> Dict[str, Any]:
        return await asyncio.to_thread(self.chat_response_sync, payload, timeout=timeout)

    def chat_text_with_response_formats_sync(
        self,
        base_payload: Dict[str, Any],
        response_formats: List[Dict[str, Any]],
        *,
        timeout: Optional[int] = None,
    ) -> str:
        if not self.available():
            return ""
        endpoint_error: Optional[BaseException] = None
        model_error: Optional[BaseException] = None
        endpoint_errors: List[str] = []
        for url in self.chat_url_candidates():
            url_had_non_404 = False
            for response_format in response_formats:
                try:
                    payload = dict(base_payload)
                    payload["response_format"] = response_format
                    response = requests.post(
                        url,
                        headers=self.headers(),
                        json=payload,
                        timeout=int(timeout or self.config.LLM_TIMEOUT),
                    )
                    if response.status_code == 404:
                        body = response.text[:300].replace("\n", " ")
                        endpoint_errors.append(f"{url} -> 404 {body}")
                        if endpoint_error is None:
                            endpoint_error = RuntimeError("LLM endpoint 404: " + " | ".join(endpoint_errors))
                        continue
                    if response.status_code >= 400:
                        body = response.text[:500].replace("\n", " ")
                        raise RuntimeError(f"LLM request failed: {response.status_code} {url} {body}")
                    url_had_non_404 = True
                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices") or []
                    if choices and isinstance(choices, list):
                        content = self.extract_choice_text(choices[0])
                        if content:
                            return content
                    model_error = RuntimeError("LLM structured response missing valid content")
                except Exception as exc:
                    model_error = exc
                    continue
            if url_had_non_404:
                break
        raise model_error or endpoint_error or RuntimeError("LLM structured generation failed")

    async def chat_text_with_response_formats(
        self,
        base_payload: Dict[str, Any],
        response_formats: List[Dict[str, Any]],
        *,
        timeout: Optional[int] = None,
    ) -> str:
        return await asyncio.to_thread(
            self.chat_text_with_response_formats_sync,
            base_payload,
            response_formats,
            timeout=timeout,
        )
