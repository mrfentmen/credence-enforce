"""
credence/providers.py
=====================
Provider-agnostic client factory.

Exposes an Anthropic-compatible interface (client.messages.create)
so eval scripts can swap providers with a single --provider flag.

Supported providers:
    anthropic  — Anthropic API (ANTHROPIC_API_KEY)
    hf         — HuggingFace Inference router (HF_TOKEN)
    groq       — Groq API, genuinely free tier (GROQ_API_KEY)
"""
from __future__ import annotations
import os
import time
import json


def _requests():
    """The ``requests`` module, imported on first use.

    Mirrors how ``anthropic`` is handled in :func:`make_client`: the SDK a
    provider needs is imported when that provider is used, not when this module
    is imported. This import used to sit at module scope while ``requests`` was
    not declared in pyproject.toml at all, so ``import credence.providers``
    raised ModuleNotFoundError on a clean install — which is what made the
    legacy suite's S22 and S23 suites blanket-fail 14 checks in CI. It is now
    declared in the ``api`` extra, and the failure names it if it is absent.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "The hf and groq providers need `requests`. Install it with: "
            "pip install 'credence-enforce[api]'"
        ) from exc
    return requests

# ---------------------------------------------------------------------------
# HuggingFace model names
# ---------------------------------------------------------------------------
HF_COMPRESS_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
HF_DOWNSTREAM_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

_HF_BASE   = "https://router.huggingface.co/v1/chat/completions"
_HF_TOKEN  = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Groq model names  (free tier: 14,400 req/day, 30 req/min)
# ---------------------------------------------------------------------------
GROQ_COMPRESS_MODEL    = "llama-3.1-8b-instant"
GROQ_DOWNSTREAM_MODEL  = "llama-3.1-8b-instant"   # 14,400 req/day free; 70B only 1,000/day

_GROQ_BASE  = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_TOKEN = os.environ.get("GROQ_API_KEY", "")

_RETRY_CODES  = {503, 429}
_MAX_RETRIES  = 5
_RETRY_SLEEP  = 15


# ---------------------------------------------------------------------------
# Minimal Anthropic-response shim so callers can do resp.content[0].text
# ---------------------------------------------------------------------------
class _Content:
    def __init__(self, text: str):
        self.text = text

class _Response:
    def __init__(self, text: str):
        self.content = [_Content(text)]


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible client (used for both HF and Groq)
# ---------------------------------------------------------------------------
class _OAIClient:
    """
    Drop-in for anthropic.Anthropic() using any OpenAI-compatible endpoint.
    """

    def __init__(self, base_url: str, token: str, provider_tag: str,
                 rate_sleep: float = 0.6):
        self._base      = base_url
        self._token     = token
        self._tag       = provider_tag
        self._rate_sleep = rate_sleep
        self.messages   = self

    def create(
        self,
        model: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 300,
        **_kwargs,
    ) -> _Response:
        requests = _requests()
        oai_msgs: list[dict] = []
        if system:
            oai_msgs.append({"role": "system", "content": system})
        oai_msgs.extend(messages)

        payload = {"model": model, "messages": oai_msgs, "max_tokens": max_tokens}
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type":  "application/json",
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._base, headers=headers,
                    data=json.dumps(payload), timeout=120,
                )
                if resp.status_code in _RETRY_CODES:
                    wait = _RETRY_SLEEP * attempt
                    try:
                        detail = resp.json().get("error", {}).get("message", "")[:120]
                    except Exception:
                        detail = resp.text[:80]
                    print(f"    [{self._tag}] {resp.status_code} ({detail}) — "
                          f"retry {attempt}/{_MAX_RETRIES} in {wait}s …")
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    # Non-retryable client error — print body and fail fast
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = resp.text[:200]
                    raise RuntimeError(
                        f"{self._tag} {resp.status_code}: {detail}"
                    )
                text = resp.json()["choices"][0]["message"]["content"].strip()
                time.sleep(self._rate_sleep)
                return _Response(text)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(f"{self._tag} network error: {exc}")
                print(f"    [{self._tag}] network error, retry {attempt}/{_MAX_RETRIES} …")
                time.sleep(_RETRY_SLEEP)

        raise RuntimeError(f"{self._tag} request failed after all retries")


def HFClient(token: str | None = None) -> _OAIClient:
    return _OAIClient(_HF_BASE, token or _HF_TOKEN, "hf", rate_sleep=0.6)

def GroqClient(token: str | None = None) -> _OAIClient:
    return _OAIClient(_GROQ_BASE, token or _GROQ_TOKEN, "groq", rate_sleep=5.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_client(provider: str = "anthropic"):
    """
    provider: 'anthropic' | 'hf' | 'groq'
    Returns a client with .messages.create() matching Anthropic SDK shape.
    """
    if provider == "hf":
        return HFClient()
    if provider == "groq":
        key = _GROQ_TOKEN
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com")
        return GroqClient(key)

    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key and api_key != "your-key-here":
            return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        pass

    raise EnvironmentError(
        "No API key found. Use --provider groq (free) or --provider hf."
    )
