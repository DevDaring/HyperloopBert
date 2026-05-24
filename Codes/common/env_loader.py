"""
env_loader.py
-------------
Load and resolve environment variables for the HyperloopBert pipeline.

The .env file lives at:  d:/PhD/HyperloopBert/Codes/.env
This module is at:       d:/PhD/HyperloopBert/Codes/common/env_loader.py

Alias mapping allows caller code to use canonical short names (e.g., GCP_KEY1)
while the actual .env uses the original long names (e.g., GEMINI_API_KEY_1).
All lookups are case-insensitive.

Key values are NEVER logged.

Usage:
    from common.env_loader import env_loader
    hf_token = env_loader.get("HF_TOKEN")
    keys = env_loader.get_provider_keys("gemini")
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alias map: canonical alias (lower) -> real .env key (lower)
# This maps alternative short names used in code to the actual key names stored
# in .env.  The mapping is bidirectional: the real key is always accessible
# by its own name too.
# ---------------------------------------------------------------------------
_ALIAS_MAP: Dict[str, str] = {
    # HuggingFace
    "hf_key":            "huggingface_token",
    "hf_token":          "huggingface_token",
    # Gemini
    "gcp_key1":          "gemini_api_key_1",
    "gcp_key2":          "gemini_api_key_2",
    "gcp_key3":          "gemini_api_key_3",
    "gcp_key4":          "gemini_api_key_4",
    # DeepSeek
    "deepseek_key1":     "deepseek_api_key_1",
    "deepseek_key2":     "deepseek_api_key_2",
    # OpenRouter
    "openrouter_key1":   "openrouter_api_key_1",
    "openrouter_key2":   "openrouter_api_key_2",
    # Mistral  (note: real .env has mixed case Mistral_API_KEY2)
    "mistral_key1":      "mistral_api_key1",
    "mistral_key2":      "mistral_api_key2",
}

# ---------------------------------------------------------------------------
# Provider -> ordered list of canonical aliases for that provider's API keys
# ---------------------------------------------------------------------------
_PROVIDER_KEY_ALIASES: Dict[str, List[str]] = {
    "gemini":      ["gcp_key1", "gcp_key2", "gcp_key3", "gcp_key4"],
    "deepseek":    ["deepseek_key1", "deepseek_key2"],
    "mistral":     ["mistral_key1", "mistral_key2"],
    "openrouter":  ["openrouter_key1", "openrouter_key2"],
    "hf":          ["hf_token"],
    "openai":      [],  # no OpenAI keys in this project
}

# ---------------------------------------------------------------------------
# Default values for optional config keys
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, str] = {
    "gemini_model_name":         "gemini-2.5-flash-lite",
    "deepseek_api_base_url":     "https://api.deepseek.com/v1",
    "openrouter_api_base_url":   "https://openrouter.ai/api/v1",
}


def _locate_env_file() -> Optional[Path]:
    """
    Walk up from this file's directory to find the .env file.
    Expected structure:
        Codes/.env
        Codes/common/env_loader.py  <-- this file
    """
    current = Path(__file__).resolve().parent
    # Try two levels up to be safe.
    for _ in range(3):
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        current = current.parent
    return None


def _parse_env_file(path: Path) -> Dict[str, str]:
    """
    Parse a .env file into a dict of {lower_key: value}.

    Rules:
    - Lines starting with '#' are comments.
    - Empty lines are skipped.
    - Key=value pairs; value may be optionally quoted (single or double).
    - Whitespace around key and value is stripped.
    - Keys are lowercased for case-insensitive lookup.
    - Empty values are stored as empty strings (callers treat them as missing).
    """
    result: Dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key_part, _, value_part = line.partition("=")
                key = key_part.strip().lower()
                value = value_part.strip()
                # Strip optional surrounding quotes.
                if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                value = value.strip()
                result[key] = value
    except OSError as exc:
        logger.error("Failed to read .env file at %s: %s", path, exc)
    return result


class EnvLoader:
    """
    Case-insensitive environment variable resolver with alias support.

    All key lookups go through this resolution chain:
    1. Normalize the requested key to lowercase.
    2. Check the alias map; if found, resolve to the real key.
    3. Look up the real key in the loaded env dict.
    4. Fall back to os.environ (also lowercased).
    5. If still not found, check _DEFAULTS.

    Key VALUES are never logged.
    """

    def __init__(self, env_dict: Dict[str, str]) -> None:
        # All keys stored lowercase.
        self._env: Dict[str, str] = env_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, key: str) -> str:
        """
        Resolve a possibly aliased key to the actual .env key (lowercase).
        """
        lower = key.strip().lower()
        return _ALIAS_MAP.get(lower, lower)

    def _raw_get(self, real_key: str) -> Optional[str]:
        """
        Look up a real (already resolved) lowercase key.
        Returns None if not found or if the value is empty.
        """
        value = self._env.get(real_key)
        if value is None:
            # Fall back to os.environ (case-insensitive via lowercase scan).
            value = os.environ.get(real_key.upper()) or os.environ.get(real_key)
        if value is None:
            value = _DEFAULTS.get(real_key)
        if value is not None:
            value = value.strip()
            if value == "":
                return None
        return value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a value by key (or alias), case-insensitively.

        Parameters
        ----------
        key : str
            The environment variable name or alias to look up.

        Returns
        -------
        str or None
            The value, or None if not set / empty.
        """
        real_key = self._resolve_key(key)
        return self._raw_get(real_key)

    def get_provider_keys(self, provider: str) -> List[str]:
        """
        Return ordered list of non-empty API key values for the given provider.

        Parameters
        ----------
        provider : str
            One of: 'gemini', 'deepseek', 'mistral', 'openrouter', 'hf', 'openai'.

        Returns
        -------
        list of str
            Non-empty key values in the order defined for the provider.

        Raises
        ------
        ValueError
            If an unknown provider name is given.
        """
        provider_lower = provider.strip().lower()
        if provider_lower not in _PROVIDER_KEY_ALIASES:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Valid providers: {sorted(_PROVIDER_KEY_ALIASES.keys())}"
            )
        aliases = _PROVIDER_KEY_ALIASES[provider_lower]
        keys: List[str] = []
        for alias in aliases:
            value = self.get(alias)
            if value:
                keys.append(value)
        return keys

    def is_provider_available(self, provider: str) -> bool:
        """
        Return True if at least one non-empty key exists for the provider.

        Parameters
        ----------
        provider : str
            Provider name (see get_provider_keys for valid values).

        Returns
        -------
        bool
        """
        try:
            return len(self.get_provider_keys(provider)) > 0
        except ValueError:
            return False

    def log_provider_summary(self, log: logging.Logger) -> None:
        """
        Log the number of available keys per provider. Values are never logged.

        Parameters
        ----------
        log : logging.Logger
            Logger to write to.
        """
        log.info("Provider key availability summary:")
        for provider in sorted(_PROVIDER_KEY_ALIASES.keys()):
            try:
                count = len(self.get_provider_keys(provider))
                available = count > 0
                log.info(
                    "  provider=%-12s  keys_found=%d  available=%s",
                    provider,
                    count,
                    available,
                )
            except ValueError as exc:
                log.warning("  provider=%-12s  error=%s", provider, exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def get_env_loader() -> EnvLoader:
    """
    Locate the .env file, parse it, and return a cached EnvLoader instance.

    The .env file is searched by walking up from this module's directory.
    A warning is logged if the file cannot be found; an empty dict is used
    in that case so the loader can still fall back to os.environ and defaults.

    Returns
    -------
    EnvLoader
    """
    env_path = _locate_env_file()
    if env_path is None:
        logger.warning(
            "No .env file found by walking up from %s. "
            "Falling back to os.environ and defaults only.",
            Path(__file__).resolve().parent,
        )
        env_dict: Dict[str, str] = {}
    else:
        logger.debug("Loading .env from: %s", env_path)
        env_dict = _parse_env_file(env_path)
        logger.debug(
            ".env loaded: %d keys found (values not logged)", len(env_dict)
        )

    return EnvLoader(env_dict)


# Cache at module import time.
env_loader: EnvLoader = get_env_loader()
