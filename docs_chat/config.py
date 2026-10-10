"""Settings, read from environment variables (and a local .env file if present).

Every knob of the app lives here so that the README can document it in one table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines, optional quotes, inline ``# comments``.

    Never overrides real environment variables.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- LLM provider -----------------------------------------------------
    # "openai_compat": any OpenAI-compatible endpoint (Groq, OpenRouter, Mistral, Gemini's
    # OpenAI endpoint, OpenAI) via the `openai` SDK; "anthropic": Claude via the `anthropic` SDK.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai_compat"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    # No universal default model for openai_compat (set LLM_MODEL); the anthropic provider
    # falls back to claude-haiku-5-5 (see __post_init__).
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_models_available: tuple[str, ...] = ("claude-haiku-5-5", "claude-sonnet-5-5")  # anthropic dropdown (M3)
    # --- embeddings -------------------------------------------------------
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "e5"))  # "e5" | "hash"
    e5_model_name: str = field(
        default_factory=lambda: _env("E5_MODEL_NAME", "intfloat/multilingual-e5-small")
    )
    # --- chunking ---------------------------------------------------------
    chunk_words: int = field(default_factory=lambda: int(_env("CHUNK_WORDS", "250")))
    chunk_overlap_words: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP_WORDS", "50")))
    # --- retrieval --------------------------------------------------------
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    candidates: int = field(default_factory=lambda: int(_env("CANDIDATES", "20")))
    rrf_k: int = field(default_factory=lambda: int(_env("RRF_K", "60")))
    weak_retrieval_threshold: float = field(
        default_factory=lambda: float(_env("WEAK_RETRIEVAL_THRESHOLD", "0.75"))
    )
    # --- limits (protect the public demo) ----------------------------------
    max_files: int = field(default_factory=lambda: int(_env("MAX_FILES", "10")))
    max_file_mb: int = field(default_factory=lambda: int(_env("MAX_FILE_MB", "20")))
    max_total_pages: int = field(default_factory=lambda: int(_env("MAX_TOTAL_PAGES", "300")))
    max_questions_per_session: int = field(
        default_factory=lambda: int(_env("MAX_QUESTIONS_PER_SESSION", "20"))
    )
    max_daily_usd: float = field(default_factory=lambda: float(_env("MAX_DAILY_USD", "2")))
    budget_file: Path = field(default_factory=lambda: Path(_env("BUDGET_FILE", ".budget.json")))
    session_ttl_hours: float = field(default_factory=lambda: float(_env("SESSION_TTL_HOURS", "2")))
    # --- paths --------------------------------------------------------------
    sample_docs_dir: Path = field(
        default_factory=lambda: Path(_env("SAMPLE_DOCS_DIR", "eval/sample_docs"))
    )
    sessions_dir: Path = field(default_factory=lambda: Path(_env("SESSIONS_DIR", "sessions")))  # per-session uploads + index

    def __post_init__(self) -> None:
        if not self.llm_model and self.llm_provider == "anthropic":
            object.__setattr__(self, "llm_model", "claude-haiku-5-5")  # frozen dataclass


settings = Settings()
