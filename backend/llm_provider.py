import os
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("intellirag.llm")


class BaseLLMProvider(ABC):
    """Abstract modular interface for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the provider is configured and reachable."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generates text from the LLM given a prompt and optional system instructions."""
        pass


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini Flash LLM Provider using direct REST with fallback candidate models:
    Prioritizes models active on v1beta: gemini-3.5-flash-lite, gemini-3.5-flash, gemini-3.8-flash, gemini-flash-latest.
    """

    CANDIDATE_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        configured_model = os.getenv("GEMINI_MODEL", "").strip()
        self.model_name = (
            model_name
            or (configured_model if configured_model and "2.5" not in configured_model else self.CANDIDATE_MODELS[0])
        )
        self._active_model = self.model_name

    @property
    def name(self) -> str:
        return f"Gemini ({self._active_model})"

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(
                "GeminiProvider is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        models_to_try = [self._active_model] + [m for m in self.CANDIDATE_MODELS if m != self._active_model]
        last_error = None

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": max(0.0, min(1.0, temperature)),
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=14.0) as client:
                        resp = client.post(url, json=payload)

                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts).strip()
                            if text:
                                self._active_model = model
                                return text

                    # Handle 404
                    if resp.status_code == 404:
                        logger.warning(f"Model {model} returned 404. Trying next model...")
                        break
                    elif resp.status_code in (429, 503):
                        wait_sec = 3.0 * (attempt + 1)
                        logger.info(f"Model {model} rate limited ({resp.status_code}). Backing off for {wait_sec}s...")
                        time.sleep(wait_sec)
                        continue
                    else:
                        last_error = f"API status {resp.status_code}: {resp.text[:180]}"
                        break
                except Exception as exc:
                    last_error = exc
                    logger.warning(f"Request to {model} failed (attempt {attempt+1}): {exc}")
                    time.sleep(2.0)

        raise RuntimeError(f"Gemini generation failed on all candidate models. Last error: {last_error}")


class LocalFallbackProvider(BaseLLMProvider):
    """
    Fallback provider using local CPU seq2seq FLAN-T5-small when no API key is provided.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "Local CPU (FLAN-T5-small Fallback)"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 150,
    ) -> str:
        from backend.embedding_utils import generate_with_flan
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"Instructions: {system_prompt}\n\n{prompt}"
        return generate_with_flan(full_prompt, max_new_tokens=min(max_tokens, 128))


_ACTIVE_PROVIDER: Optional[BaseLLMProvider] = None


def get_llm_provider(force_refresh: bool = False) -> BaseLLMProvider:
    """
    Factory that returns the configured LLM provider according to environment variables.
    Defaults to GeminiProvider if GEMINI_API_KEY is present, else LocalFallbackProvider.
    """
    global _ACTIVE_PROVIDER
    if _ACTIVE_PROVIDER is not None and not force_refresh:
        return _ACTIVE_PROVIDER

    load_dotenv()
    provider_type = os.getenv("LLM_PROVIDER", "").lower().strip()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if provider_type == "local" or (not gemini_key and provider_type != "gemini"):
        logger.info("Initializing LocalFallbackProvider (FLAN-T5-small).")
        _ACTIVE_PROVIDER = LocalFallbackProvider()
    else:
        logger.info("Initializing GeminiProvider with configured API key.")
        _ACTIVE_PROVIDER = GeminiProvider(api_key=gemini_key)

    return _ACTIVE_PROVIDER
