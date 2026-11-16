"""LLM provider layer: Ollama (local), optional OpenAI-compatible API, or none.

Design constraints:

* provider-specific HTTP details never leak into retrieval, reasoning or routes;
* every call has a timeout and bounded retries with linear backoff;
* a missing/failing provider degrades the copilot to retrieval-only answers
  instead of crashing the quantitative API;
* no secrets are ever logged or returned from ``describe()``.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator

import requests

from src.ai.config import AIConfig, get_ai_config
from src.ai.errors import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_CONNECT_TIMEOUT_S = 2.0
_OLLAMA_TAGS_PATH = "/api/tags"
_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"


class LLMProvider(ABC):
    """Minimal chat-completion interface (single-turn)."""

    name: str = "base"
    supports_streaming: bool = False

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Return the model's text answer or raise a typed ``LLMError``."""

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> Iterator[str]:
        """Yield text deltas as the model generates them.

        Connection/first-chunk failures raise typed ``LLMError``s; once
        streaming has started a protocol corruption raises
        :class:`ProviderResponseError`. Providers without streaming support
        fall back to a single-delta answer.
        """
        yield self.complete(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap liveness probe; never raises."""

    def describe(self) -> dict[str, Any]:
        """Public, secret-free status payload."""
        return {
            "provider": self.name,
            "available": self.is_available(),
            "supports_streaming": self.supports_streaming,
        }


class NullProvider(LLMProvider):
    """Explicitly disabled provider (retrieval-only mode)."""

    name = "none"

    def complete(self, prompt, *, system=None, max_tokens=None, temperature=None, timeout_s=None):
        raise ProviderUnavailableError(
            "no LLM provider is configured - set ALPHAFORGE_LLM_PROVIDER=ollama "
            "with a running Ollama server, or configure a hosted provider"
        )

    def stream(self, prompt, *, system=None, max_tokens=None, temperature=None, timeout_s=None):
        raise ProviderUnavailableError(
            "no LLM provider is configured - streaming needs Ollama or a hosted provider"
        )
        yield ""  # pragma: no cover - makes this a generator; unreachable

    def is_available(self) -> bool:
        return False


class OllamaProvider(LLMProvider):
    """Local Ollama server via its HTTP API (no SDK dependency)."""

    name = "ollama"
    supports_streaming = True

    def __init__(self, model: str, base_url: str, cfg: AIConfig):
        self.model = model
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self._cfg = cfg

    def is_available(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}{_OLLAMA_TAGS_PATH}", timeout=_CONNECT_TIMEOUT_S
            )
            return response.ok
        except requests.RequestException:
            return False

    def complete(self, prompt, *, system=None, max_tokens=None, temperature=None, timeout_s=None):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages(prompt, system),
            "stream": False,
            "options": {
                "temperature": float(
                    temperature if temperature is not None else self._cfg.llm_temperature
                ),
                "num_predict": int(max_tokens or self._cfg.llm_max_tokens),
            },
        }
        data = _post_with_retries(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self._timeout(timeout_s),
            max_retries=self._cfg.llm_max_retries,
            backoff=self._cfg.llm_retry_backoff_s,
            what=f"ollama:{self.model}",
        )
        message = (data.get("message") or {}).get("content")
        if not isinstance(message, str) or not message.strip():
            raise ProviderResponseError("ollama returned an empty/malformed response")
        return message.strip()

    def _timeout(self, override: float | None) -> float:
        return float(override if override is not None else self._cfg.llm_timeout_s)

    def stream(self, prompt, *, system=None, max_tokens=None, temperature=None, timeout_s=None):
        """Stream NDJSON chunks from ``POST /api/chat`` (``stream: true``)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages(prompt, system),
            "stream": True,
            "options": {
                "temperature": float(
                    temperature if temperature is not None else self._cfg.llm_temperature
                ),
                "num_predict": int(max_tokens or self._cfg.llm_max_tokens),
            },
        }
        timeout = self._timeout(timeout_s)
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
                stream=True,
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(
                f"ollama:{self.model} timed out after {timeout:.0f}s"
            ) from exc
        except requests.RequestException as exc:
            raise ProviderUnavailableError(
                f"ollama:{self.model} is unreachable: {exc}"
            ) from exc
        if not response.ok:
            detail = _safe_detail(response)
            if response.status_code in (401, 403):
                raise ProviderUnavailableError(f"ollama:{self.model} rejected the request ({detail})")
            raise ProviderUnavailableError(
                f"ollama:{self.model} answered HTTP {response.status_code}: {detail}"
            )
        got_any = False
        try:
            for raw in response.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ProviderResponseError(
                        "ollama stream contained a malformed chunk"
                    ) from exc
                if not isinstance(chunk, dict):
                    raise ProviderResponseError("ollama stream chunk was not an object")
                if chunk.get("error"):
                    raise ProviderUnavailableError(
                        f"ollama:{self.model} stream error: {str(chunk['error'])[:200]}"
                    )
                content = (chunk.get("message") or {}).get("content")
                if isinstance(content, str) and content:
                    got_any = True
                    yield content
                if chunk.get("done"):
                    break
        except requests.RequestException as exc:
            raise ProviderUnavailableError(
                f"ollama:{self.model} stream was interrupted: {exc}"
            ) from exc
        finally:
            response.close()
        if not got_any:
            raise ProviderResponseError("ollama stream ended without any content")


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible ``/chat/completions`` endpoint (hosted or self-hosted)."""

    name = "openai"
    supports_streaming = True

    def __init__(self, model: str, base_url: str, api_key: str, cfg: AIConfig):
        self.model = model
        self.base_url = (base_url or _OPENAI_DEFAULT_BASE).rstrip("/")
        self._api_key = api_key
        self._cfg = cfg

    def is_available(self) -> bool:
        return bool(self._api_key)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "available": self.is_available(),
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self._api_key),
        }

    def complete(self, prompt, *, system=None, max_tokens=None, temperature=None, timeout_s=None):
        if not self._api_key:
            raise ProviderUnavailableError(
                "hosted provider selected but no API key is set "
                "(ALPHAFORGE_LLM_API_KEY or OPENAI_API_KEY)"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages(prompt, system),
            "temperature": float(temperature if temperature is not None else self._cfg.llm_temperature),
            "max_tokens": int(max_tokens or self._cfg.llm_max_tokens),
        }
        data = _post_with_retries(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=float(timeout_s if timeout_s is not None else self._cfg.llm_timeout_s),
            max_retries=self._cfg.llm_max_retries,
            backoff=self._cfg.llm_retry_backoff_s,
            what=f"openai:{self.model}",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError("hosted provider returned a malformed payload") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("hosted provider returned an empty answer")
        return content.strip()

    def stream(self, prompt, *, system=None, max_tokens=None, temperature=None, timeout_s=None):
        """Stream SSE chunks from ``/chat/completions`` (``stream: true``)."""
        if not self._api_key:
            raise ProviderUnavailableError(
                "hosted provider selected but no API key is set "
                "(ALPHAFORGE_LLM_API_KEY or OPENAI_API_KEY)"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages(prompt, system),
            "temperature": float(temperature if temperature is not None else self._cfg.llm_temperature),
            "max_tokens": int(max_tokens or self._cfg.llm_max_tokens),
            "stream": True,
        }
        timeout = float(timeout_s if timeout_s is not None else self._cfg.llm_timeout_s)
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
                stream=True,
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(
                f"openai:{self.model} timed out after {timeout:.0f}s"
            ) from exc
        except requests.RequestException as exc:
            raise ProviderUnavailableError(f"openai:{self.model} is unreachable: {exc}") from exc
        if not response.ok:
            detail = _safe_detail(response)
            if response.status_code in (401, 403):
                raise ProviderUnavailableError(
                    f"openai:{self.model} rejected the credentials ({detail})"
                )
            raise ProviderUnavailableError(
                f"openai:{self.model} answered HTTP {response.status_code}: {detail}"
            )
        got_any = False
        try:
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    if line == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError as exc:
                    raise ProviderResponseError(
                        "hosted provider stream contained a malformed chunk"
                    ) from exc
                if not isinstance(chunk, dict):
                    raise ProviderResponseError("hosted provider stream chunk was not an object")
                if chunk.get("error"):
                    raise ProviderUnavailableError(
                        f"openai:{self.model} stream error: {str(chunk['error'])[:200]}"
                    )
                try:
                    delta = chunk["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, TypeError):
                    delta = None
                if isinstance(delta, str) and delta:
                    got_any = True
                    yield delta
        except requests.RequestException as exc:
            raise ProviderUnavailableError(
                f"openai:{self.model} stream was interrupted: {exc}"
            ) from exc
        finally:
            response.close()
        if not got_any:
            raise ProviderResponseError("hosted provider stream ended without any content")


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _post_with_retries(
    url: str,
    *,
    json: dict[str, Any],
    timeout: float,
    max_retries: int,
    backoff: float,
    what: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST with bounded retries; maps failures onto typed LLM errors."""
    attempts = max(int(max_retries), 0) + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                json=json,
                timeout=timeout,
                headers=headers or {},
            )
        except requests.Timeout:
            last_error = ProviderTimeoutError(f"{what} timed out after {timeout:.0f}s")
        except requests.RequestException as exc:
            last_error = ProviderUnavailableError(f"{what} is unreachable: {exc}")
        else:
            if response.ok:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ProviderResponseError(f"{what} returned non-JSON output") from exc
                if not isinstance(payload, dict):
                    raise ProviderResponseError(f"{what} returned a non-object payload")
                return payload
            detail = _safe_detail(response)
            if response.status_code in (401, 403):
                raise ProviderUnavailableError(f"{what} rejected the credentials ({detail})")
            last_error = ProviderUnavailableError(
                f"{what} answered HTTP {response.status_code}: {detail}"
            )
        if attempt < attempts - 1:
            sleep_for = backoff * (attempt + 1)
            log.warning("%s failed (%s) - retrying in %.1fs", what, last_error, sleep_for)
            time.sleep(sleep_for)
    assert last_error is not None
    raise last_error


def _safe_detail(response: requests.Response) -> str:
    """Short, secret-free error detail from a provider response."""
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "")[:200]
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:200]
        if err:
            return str(err)[:200]
    return json.dumps(payload)[:200]


def build_provider(cfg: AIConfig | None = None) -> LLMProvider:
    """Resolve the configured provider (``auto`` probes Ollama, then hosted)."""
    cfg = cfg or get_ai_config()
    choice = (cfg.llm_provider or "auto").lower()
    if choice == "none":
        return NullProvider()
    if choice == "ollama":
        return OllamaProvider(cfg.llm_model, cfg.ollama_base_url, cfg)
    if choice == "openai":
        return OpenAIProvider(cfg.llm_model, cfg.llm_base_url, cfg.llm_api_key, cfg)
    # auto: prefer local Ollama when it answers, else hosted when configured.
    ollama = OllamaProvider(cfg.llm_model, cfg.ollama_base_url, cfg)
    if ollama.is_available():
        log.info("auto-selected local Ollama at %s", cfg.ollama_base_url)
        return ollama
    if cfg.llm_api_key or cfg.llm_base_url:
        log.info("auto-selected hosted OpenAI-compatible provider")
        return OpenAIProvider(cfg.llm_model, cfg.llm_base_url, cfg.llm_api_key, cfg)
    log.info("no LLM provider reachable - copilot runs in retrieval-only mode")
    return NullProvider()


def provider_report(cfg: AIConfig | None = None) -> dict[str, Any]:
    """Health payload describing every provider without leaking secrets."""
    cfg = cfg or get_ai_config()
    ollama = OllamaProvider(cfg.llm_model, cfg.ollama_base_url, cfg)
    return {
        "configured": cfg.llm_provider,
        "ollama": {
            "base_url": cfg.ollama_base_url,
            "model": cfg.llm_model,
            "reachable": ollama.is_available(),
            "supports_streaming": True,
        },
        "hosted": {
            "provider": "openai-compatible",
            "api_key_configured": bool(cfg.llm_api_key),
            "base_url": cfg.llm_base_url or None,
            "supports_streaming": True,
        },
    }
