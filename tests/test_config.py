from __future__ import annotations

import os
from pathlib import Path

from docs_chat.config import _load_dotenv

KEYS = ("DOCS_CHAT_T_PLAIN", "DOCS_CHAT_T_QUOTED", "DOCS_CHAT_T_COMMENTED", "DOCS_CHAT_T_IGNORED", "DOCS_CHAT_T_EMPTY")


def test_dotenv_loader_handles_quotes_comments_and_never_overrides_env(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "DOCS_CHAT_T_PLAIN=plain\n"
        'DOCS_CHAT_T_QUOTED="quoted # not a comment"\n'
        "DOCS_CHAT_T_COMMENTED=e5                    # e5 | hash\n"
        "# DOCS_CHAT_T_IGNORED=ignored\n"
        "DOCS_CHAT_T_EMPTY=\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCS_CHAT_T_PLAIN", "from-real-env")
    try:
        _load_dotenv(env)
        assert os.environ["DOCS_CHAT_T_PLAIN"] == "from-real-env"
        assert os.environ["DOCS_CHAT_T_QUOTED"] == "quoted # not a comment"
        assert os.environ["DOCS_CHAT_T_COMMENTED"] == "e5"
        assert "DOCS_CHAT_T_IGNORED" not in os.environ
        assert os.environ["DOCS_CHAT_T_EMPTY"] == ""
    finally:
        for key in KEYS:
            os.environ.pop(key, None)


def test_dotenv_loader_ignores_missing_file(tmp_path: Path):
    _load_dotenv(tmp_path / "missing.env")  # no error
