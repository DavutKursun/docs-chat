from __future__ import annotations

import os
from pathlib import Path

from docs_chat.config import Settings, _load_dotenv, on_space

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


def test_public_demo_limits_apply_on_a_space(monkeypatch):
    for key in ("SPACE_ID", "MAX_FILES", "MAX_FILE_MB", "MAX_TOTAL_PAGES"):
        monkeypatch.delenv(key, raising=False)
    local = Settings()
    assert (local.max_files, local.max_file_mb, local.max_total_pages) == (10, 20, 300) and not on_space()
    monkeypatch.setenv("SPACE_ID", "someone/docs-chat")
    demo = Settings()
    assert (demo.max_files, demo.max_file_mb, demo.max_total_pages) == (3, 10, 60) and demo.on_space
    assert demo.max_questions_per_session == 20 and demo.max_daily_answers == 150
    monkeypatch.setenv("MAX_FILES", "5")  # an explicit variable still wins
    assert Settings().max_files == 5
