"""
llm_utils.py
------------
Multi-provider JSON-first LLM client for the HyperloopBert pipeline.

Design contract
---------------
NO automatic cross-tier fallback.

The caller selects a provider explicitly via the `provider` argument.
If that provider fails (all retries exhausted, auth error, quota exceeded),
the error is surfaced to the caller as an exception. This module NEVER
silently switches to a different provider tier.

Rationale: silent provider switching corrupts reproducibility. Each
provider has different capabilities, pricing, and rate limits. The caller
must be explicit about which tier is acceptable for a given task.

Within-provider retry policy
----------------------------
Transient errors (HTTP 429, 5xx, timeout) trigger a retry using the next
available key for that provider (round-robin within the provider's key list).
A maximum of min(3, num_keys) attempts are made before raising.

JSON enforcement
----------------
- Gemini: responseMimeType="application/json" in generation config.
- OpenAI-compatible: response_format={"type": "json_object"}.
- The JSON schema (if provided) is embedded in the prompt text as a backstop
  so the model knows what fields are expected even if the MIME type is ignored.
- Response parsing: strip markdown fences -> find first balanced {...} block ->
  json.loads() -> optionally validate required keys.
- On any parse failure: log raw response to logs/llm_parse_failures.log and
  return {"needs_review": True, "raw": "<truncated response>"}.

Providers
---------
| Provider    | Judge tier | Model                    | Keys                |
|-------------|------------|--------------------------|---------------------|
| deepseek    | PRIMARY    | deepseek-chat            | DEEPSEEK_API_KEY_1/2 |
| mistral     | SECONDARY  | mistral-small-latest     | MISTRAL_API_KEY1/2  |
| openrouter  | TERTIARY   | openai/gpt-4o-mini       | OPENROUTER_API_KEY_1/2 |
| gemini      | DISABLED   | -- (never called)        | --                  |

JUDGEMENT POLICY (project requirement): Gemini is NEVER used. For any
judgement/extraction task call `call_judge()`, which tries DeepSeek (primary),
then Mistral (secondary), then OpenRouter/gpt-4o-mini (tertiary), round-robining
keys WITHIN each provider. `call_llm(provider=...)` keeps its strict no-fallback
contract for callers that must pin one provider; `call_judge()` is the ordered
multi-provider path.

All HTTP calls use the requests library only. The openai Python package is
NOT required and NOT imported.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from common.env_loader import env_loader
from common.logging_setup import setup_logging

logger = setup_logging(__name__, log_dir="logs")

# ---------------------------------------------------------------------------
# Parse-failure log (append-only file for raw LLM responses that fail JSON
# parsing — kept separate from the main log so it can be grepped easily).
# ---------------------------------------------------------------------------
_PARSE_FAIL_LOG_PATH = os.path.join("logs", "llm_parse_failures.log")


def _log_parse_failure(provider: str, raw: str) -> None:
    """Append a raw LLM response to the parse-failure log file."""
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    entry = (
        f"\n{'=' * 80}\n"
        f"TIMESTAMP : {timestamp}\n"
        f"PROVIDER  : {provider}\n"
        f"RAW (first 4000 chars):\n{raw[:4000]}\n"
    )
    try:
        with open(_PARSE_FAIL_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(entry)
    except OSError as exc:
        logger.error("Could not write parse-failure log: %s", exc)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```json ... ```)."""
    text = text.strip()
    # Remove opening fence with optional language tag.
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    # Remove closing fence.
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def _find_first_json_object(text: str) -> Optional[str]:
    """
    Find the first balanced {...} block in `text`.
    Returns the substring or None if no balanced block is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_response_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract a JSON object from an LLM text response.
    Returns the parsed dict or None on failure.
    """
    cleaned = _strip_markdown_fences(text)
    # Try the whole cleaned text first.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try finding the first balanced {...} block.
    block = _find_first_json_object(cleaned)
    if block:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
    return None


def _build_schema_instruction(schema: Optional[Dict[str, Any]]) -> str:
    """
    Produce a text block instructing the model to conform to the given schema.
    If schema is None, returns a generic JSON instruction.
    """
    if schema is None:
        return (
            "\n\nIMPORTANT: Respond with a single valid JSON object only. "
            "Do not include markdown fences, explanatory text, or trailing commas."
        )
    schema_str = json.dumps(schema, indent=2)
    return (
        f"\n\nIMPORTANT: Respond with a single valid JSON object only. "
        f"The response MUST conform to this schema:\n{schema_str}\n"
        f"Do not include markdown fences, explanatory text, or trailing commas."
    )


# ---------------------------------------------------------------------------
# Provider call implementations
# ---------------------------------------------------------------------------

def _call_gemini(
    prompt: str,
    keys: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Call the Gemini API using the google.generativeai library if available,
    otherwise fall back to the REST endpoint.

    Rotates through `keys` on transient failure (429, 5xx, timeout).

    Returns
    -------
    str
        The raw text of the model response.

    Raises
    ------
    RuntimeError
        If all keys are exhausted or a non-retryable error is encountered.
    """
    max_attempts = min(3, len(keys))
    last_exc: Optional[Exception] = None

    for attempt_idx in range(max_attempts):
        key = keys[attempt_idx % len(keys)]
        logger.debug(
            "Gemini call: model=%s attempt=%d/%d key_index=%d",
            model, attempt_idx + 1, max_attempts, attempt_idx,
        )
        try:
            return _gemini_via_library_or_rest(
                api_key=key,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except _RetryableError as exc:
            logger.warning(
                "Gemini transient error on attempt %d/%d: %s -- rotating key",
                attempt_idx + 1, max_attempts, exc,
            )
            last_exc = exc
            time.sleep(1.5 * (attempt_idx + 1))
        except _FatalError:
            raise

    raise RuntimeError(
        f"Gemini provider failed after {max_attempts} attempts. "
        f"Last error: {last_exc}"
    )


def _gemini_via_library_or_rest(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Try google.generativeai library first; fall back to REST if not installed.
    """
    try:
        import google.generativeai as genai  # type: ignore
        return _gemini_via_library(
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            genai=genai,
        )
    except ImportError:
        logger.debug(
            "google.generativeai not installed; using REST API for Gemini."
        )
        return _gemini_via_rest(
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def _gemini_via_library(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    genai: Any,
) -> str:
    """Call Gemini using the google.generativeai Python library."""
    import google.generativeai.types as genai_types  # type: ignore

    try:
        genai.configure(api_key=api_key)
        generation_config = genai_types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        model_obj = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )
        response = model_obj.generate_content(prompt)
        return response.text
    except Exception as exc:
        _classify_gemini_exception(exc)


def _gemini_via_rest(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Gemini via the public REST endpoint."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
    except requests.exceptions.Timeout as exc:
        raise _RetryableError(f"Gemini REST timeout: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise _RetryableError(f"Gemini REST connection error: {exc}") from exc

    if resp.status_code in (429, 500, 502, 503, 504):
        raise _RetryableError(
            f"Gemini REST HTTP {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code == 400:
        raise _FatalError(
            f"Gemini REST bad request (400): {resp.text[:400]}"
        )
    if resp.status_code == 401 or resp.status_code == 403:
        raise _FatalError(
            f"Gemini REST auth error ({resp.status_code}): {resp.text[:200]}"
        )
    resp.raise_for_status()

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise _FatalError(
            f"Gemini REST unexpected response structure: {exc} | raw: {str(data)[:400]}"
        ) from exc


def _classify_gemini_exception(exc: Exception) -> None:
    """
    Re-raise a google.generativeai exception as _RetryableError or _FatalError.
    Always raises; never returns.
    """
    msg = str(exc)
    # Check for retryable HTTP status codes within the exception message.
    if any(code in msg for code in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED")):
        raise _RetryableError(f"Gemini library transient error: {msg}") from exc
    if any(code in msg for code in ("401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED")):
        raise _FatalError(f"Gemini library auth error: {msg}") from exc
    # For unknown errors, treat as retryable to be safe.
    raise _RetryableError(f"Gemini library error: {msg}") from exc


def _call_openai_compatible(
    prompt: str,
    keys: List[str],
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider_name: str,
) -> str:
    """
    Call an OpenAI-compatible chat completion endpoint using requests only.

    Rotates through `keys` on transient failure.

    Returns
    -------
    str
        The raw text of the assistant message.

    Raises
    ------
    RuntimeError
        If all keys are exhausted.
    """
    max_attempts = min(3, len(keys))
    last_exc: Optional[Exception] = None

    for attempt_idx in range(max_attempts):
        key = keys[attempt_idx % len(keys)]
        logger.debug(
            "%s call: model=%s attempt=%d/%d key_index=%d",
            provider_name, model, attempt_idx + 1, max_attempts, attempt_idx,
        )
        try:
            return _openai_compat_request(
                api_key=key,
                base_url=base_url,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                provider_name=provider_name,
            )
        except _RetryableError as exc:
            logger.warning(
                "%s transient error on attempt %d/%d: %s -- rotating key",
                provider_name, attempt_idx + 1, max_attempts, exc,
            )
            last_exc = exc
            time.sleep(1.5 * (attempt_idx + 1))
        except _FatalError:
            raise

    raise RuntimeError(
        f"{provider_name} provider failed after {max_attempts} attempts. "
        f"Last error: {last_exc}"
    )


def _openai_compat_request(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    provider_name: str,
) -> str:
    """Single attempt at an OpenAI-compatible /chat/completions call."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout as exc:
        raise _RetryableError(f"{provider_name} timeout: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise _RetryableError(
            f"{provider_name} connection error: {exc}"
        ) from exc

    if resp.status_code in (429, 500, 502, 503, 504):
        raise _RetryableError(
            f"{provider_name} HTTP {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code in (400, 401, 403):
        raise _FatalError(
            f"{provider_name} fatal HTTP {resp.status_code}: {resp.text[:300]}"
        )
    resp.raise_for_status()

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise _FatalError(
            f"{provider_name} unexpected response structure: {exc} | raw: {str(data)[:400]}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal sentinel exceptions
# ---------------------------------------------------------------------------

class _RetryableError(Exception):
    """Transient error; retry with next key."""


class _FatalError(Exception):
    """Non-retryable error; propagate immediately."""


# ---------------------------------------------------------------------------
# Provider configuration lookup
# ---------------------------------------------------------------------------

def _get_provider_config(provider: str) -> Dict[str, Any]:
    """
    Return the configuration dict for the requested provider.

    Raises
    ------
    ValueError
        If the provider name is unknown.
    RuntimeError
        If no API keys are available for the requested provider.
    """
    provider_lower = provider.strip().lower()

    if provider_lower == "gemini":
        keys = env_loader.get_provider_keys("gemini")
        model = env_loader.get("gemini_model_name") or "gemini-2.5-flash-lite"
        if not keys:
            raise RuntimeError(
                "No Gemini API keys found. "
                "Set GEMINI_API_KEY_1..4 in your .env file."
            )
        return {"type": "gemini", "keys": keys, "model": model}

    if provider_lower == "deepseek":
        keys = env_loader.get_provider_keys("deepseek")
        base_url = (
            env_loader.get("deepseek_api_base_url") or "https://api.deepseek.com/v1"
        )
        model = "deepseek-chat"
        if not keys:
            raise RuntimeError(
                "No DeepSeek API keys found. "
                "Set DEEPSEEK_API_KEY_1..2 in your .env file."
            )
        return {
            "type": "openai_compat",
            "keys": keys,
            "model": model,
            "base_url": base_url,
            "provider_name": "deepseek",
        }

    if provider_lower == "mistral":
        keys = env_loader.get_provider_keys("mistral")
        # Mistral uses its own endpoint; default if not set in env.
        base_url = (
            env_loader.get("mistral_api_base_url") or "https://api.mistral.ai/v1"
        )
        model = env_loader.get("mistral_model_name") or "mistral-small-latest"
        if not keys:
            raise RuntimeError(
                "No Mistral API keys found. "
                "Set MISTRAL_API_KEY1 / Mistral_API_KEY2 in your .env file."
            )
        return {
            "type": "openai_compat",
            "keys": keys,
            "model": model,
            "base_url": base_url,
            "provider_name": "mistral",
        }

    if provider_lower == "openrouter":
        keys = env_loader.get_provider_keys("openrouter")
        base_url = (
            env_loader.get("openrouter_api_base_url")
            or "https://openrouter.ai/api/v1"
        )
        # Project requirement: OpenRouter tertiary judge uses openai/gpt-4o-mini.
        model = (
            env_loader.get("openrouter_primary_model_name")
            or "openai/gpt-4o-mini"
        )
        if not keys:
            raise RuntimeError(
                "No OpenRouter API keys found. "
                "Set OPENROUTER_API_KEY_1..2 in your .env file."
            )
        return {
            "type": "openai_compat",
            "keys": keys,
            "model": model,
            "base_url": base_url,
            "provider_name": "openrouter",
        }

    raise ValueError(
        f"Unknown provider '{provider}'. "
        "Valid values: 'gemini', 'deepseek', 'mistral', 'openrouter'."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    provider: str = "gemini",
    schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Call an LLM and return a parsed JSON dict.

    Parameters
    ----------
    prompt : str
        The user-facing prompt text. A JSON schema instruction will be
        appended automatically.
    provider : str
        Which provider to use: 'gemini' (default), 'deepseek', 'mistral',
        or 'openrouter'. The caller is responsible for selecting the correct
        provider. NO cross-tier fallback is performed.
    schema : dict or None
        Optional JSON schema dict describing the expected response shape.
        If provided, it is serialised and appended to the prompt so the
        model has explicit field guidance. Required keys in the schema are
        validated on the parsed response.
    temperature : float
        Sampling temperature in [0.0, 1.0]. Default 0.1 for deterministic
        extraction / judgment tasks.
    max_tokens : int
        Maximum number of tokens to generate.

    Returns
    -------
    dict
        Parsed JSON response from the LLM.
        On parse failure, returns:
            {"needs_review": True, "raw": "<first 2000 chars of raw response>"}
        The raw response is also written to logs/llm_parse_failures.log.

    Raises
    ------
    ValueError
        Unknown provider name.
    RuntimeError
        Provider failed after all retries, or no API keys available.

    Notes
    -----
    NO cross-provider fallback. If the requested provider fails, the
    exception propagates to the caller. This is intentional: silent
    provider switching corrupts experimental reproducibility.
    """
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"temperature must be in [0.0, 2.0], got {temperature}")

    # Project policy: Gemini is disabled for every task. Enforce at the boundary
    # so no code path (or future caller) can reach a Gemini model.
    if provider.strip().lower() == "gemini":
        raise ValueError(
            "Gemini is disabled by project policy. Use call_judge() (DeepSeek -> "
            "Mistral -> OpenRouter) or an explicit non-Gemini provider."
        )

    # Build full prompt including schema instruction.
    full_prompt = prompt + _build_schema_instruction(schema)

    logger.debug(
        "call_llm: provider=%s temperature=%.2f max_tokens=%d "
        "prompt_chars=%d",
        provider, temperature, max_tokens, len(full_prompt),
    )

    config = _get_provider_config(provider)

    # Dispatch to provider-specific caller.
    raw_text: str
    if config["type"] == "gemini":
        raw_text = _call_gemini(
            prompt=full_prompt,
            keys=config["keys"],
            model=config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif config["type"] == "openai_compat":
        raw_text = _call_openai_compatible(
            prompt=full_prompt,
            keys=config["keys"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name=config["provider_name"],
        )
    else:
        raise RuntimeError(f"Internal error: unhandled config type '{config['type']}'")

    # Parse the response.
    parsed = _parse_response_text(raw_text)
    if parsed is None:
        logger.error(
            "JSON parse failure from provider=%s. Raw response written to %s.",
            provider,
            _PARSE_FAIL_LOG_PATH,
        )
        _log_parse_failure(provider, raw_text)
        return {"needs_review": True, "raw": raw_text[:2000]}

    # Optional schema key validation. `schema` is a JSON-Schema dict (with
    # top-level keys like "type"/"properties"/"required"), so the field names
    # to check for live in schema["required"] (or schema["properties"].keys()
    # as a fallback for schemas that omit "required") -- NOT the schema dict's
    # own top-level keys.
    if schema is not None:
        expected_keys = schema.get("required")
        if expected_keys is None:
            expected_keys = list(schema.get("properties", {}).keys())
        missing_keys = [k for k in expected_keys if k not in parsed]
        if missing_keys:
            logger.warning(
                "Response from provider=%s is missing expected schema keys: %s",
                provider,
                missing_keys,
            )

    logger.debug("call_llm: provider=%s parse=success keys=%s", provider, list(parsed.keys()))
    return parsed


# ---------------------------------------------------------------------------
# Judgement path: ordered multi-provider fallback (project policy)
# ---------------------------------------------------------------------------

# Ordered judge tiers. Gemini is intentionally absent and must never be added.
JUDGE_PROVIDER_ORDER: List[str] = ["deepseek", "mistral", "openrouter"]


def call_judge(
    prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Run a judgement/extraction task under the project's provider policy:
    DeepSeek (primary) -> Mistral (secondary) -> OpenRouter/gpt-4o-mini
    (tertiary), round-robining keys WITHIN each provider. Gemini is NEVER used.

    Unlike call_llm (strict, no cross-tier fallback), this DOES fall back to the
    next provider when one is unavailable (no keys) or fails after its own
    retries -- judgement tasks favour completing over pinning one provider.

    Returns the parsed JSON dict from the first provider that succeeds. If every
    tier is unavailable or fails, returns {"needs_review": True, "raw": ...}
    with the reason, so callers never crash a pipeline on judge unavailability.
    Key values are never logged.
    """
    errors = []
    for provider in JUDGE_PROVIDER_ORDER:
        if not env_loader.is_provider_available(provider):
            logger.info("call_judge: provider=%s has no keys; trying next tier.", provider)
            errors.append(f"{provider}: no keys")
            continue
        try:
            result = call_llm(prompt, provider=provider, schema=schema,
                              temperature=temperature, max_tokens=max_tokens)
            # A parse failure returns needs_review rather than raising; treat it
            # as this tier failing and fall through to the next provider.
            if isinstance(result, dict) and result.get("needs_review"):
                logger.warning("call_judge: provider=%s returned needs_review; "
                               "trying next tier.", provider)
                errors.append(f"{provider}: parse failure")
                continue
            logger.info("call_judge: provider=%s succeeded.", provider)
            return result
        except Exception as exc:  # RuntimeError (retries exhausted), auth, etc.
            logger.warning("call_judge: provider=%s failed (%s); trying next tier.",
                           provider, exc)
            errors.append(f"{provider}: {exc}")

    logger.error("call_judge: all judge tiers unavailable/failed: %s", "; ".join(errors))
    return {"needs_review": True, "raw": f"all judge tiers failed: {'; '.join(errors)}"}
