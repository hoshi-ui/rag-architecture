from typing import Any, Dict

import requests


class OcrClient:
    def __init__(self, config: Any) -> None:
        self.config = config

    def available(self) -> bool:
        return bool((self.config.OCR_SERVICE_URL or "").strip())

    def extract_image(self, image_path: str) -> Dict[str, Any]:
        service_url = (self.config.OCR_SERVICE_URL or "").strip()
        if not service_url:
            raise RuntimeError("ocr_service_url_not_configured")
        response = requests.post(
            service_url,
            headers={"Content-Type": "application/json"},
            json={
                "image_path": image_path,
                "mode": self.config.OCR_MODE,
                "lang": self.config.OCR_LANG,
            },
            timeout=self.config.OCR_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("ocr_response_invalid")
        return payload
