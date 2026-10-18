#!/usr/bin/env python
"""Headless smoke test of the Space build: start app.py the way a Space does, check /health, ask one question.

    python scripts/smoke_space.py              # uses the fake provider when LLM_API_KEY is not in the environment
    python scripts/smoke_space.py --port 7891 --timeout 120

The app is started with SPACE_ID set, so .env is ignored (secrets come from the environment only) and
the public-demo limits apply. When no LLM_API_KEY is exported, DOCS_CHAT_FAKE_PROVIDER=1 makes the app
answer from the eval set, which exercises the whole UI path without an API call. Exit code 0 = healthy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What does error code E3 mean and what should I do?"


def wait_for_health(url: str, deadline_s: float, proc: subprocess.Popen) -> dict:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError(f"app exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    payload["_startup_seconds"] = round(time.perf_counter() - t0, 1)
                    return payload
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"no /health answer within {deadline_s} s")


def text_of(content) -> str:  # Gradio 6 returns message content as a list of parts
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7891)
    parser.add_argument("--timeout", type=float, default=120, help="seconds to wait for /health")
    parser.add_argument("--log", type=Path, default=ROOT / "smoke_app.log", help="where the app's output goes")
    args = parser.parse_args(argv)

    env = {**os.environ, "SPACE_ID": os.environ.get("SPACE_ID", "local/docs-chat-smoke"),
           "GRADIO_SERVER_NAME": "127.0.0.1", "GRADIO_SERVER_PORT": str(args.port)}
    if not env.get("LLM_API_KEY"):
        env["DOCS_CHAT_FAKE_PROVIDER"] = "1"
    base = f"http://127.0.0.1:{args.port}"
    problems: list[str] = []
    with args.log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            health = wait_for_health(f"{base}/health", args.timeout, proc)
            print(f"health after {health['_startup_seconds']} s: {json.dumps({k: v for k, v in health.items() if not k.startswith('_')})}")
            if health["status"] != "ok" or health["sample_index"]["chunks"] == 0:
                problems.append("health is not ok or the sample index is empty")
            if health["limits"] != {"files": 3, "file_mb": 10, "pages": 60, "questions_per_session": 20, "answers_per_day": health["limits"]["answers_per_day"]}:
                problems.append(f"demo limits not applied: {health['limits']}")

            from gradio_client import Client  # installed with gradio

            client = Client(base, verbose=False)
            t0 = time.perf_counter()
            chat, textbox, status = client.predict(QUESTION, [], health["model_label"], api_name="/ask")
            seconds = time.perf_counter() - t0
            final = text_of(chat[-1]["content"]) if chat else ""
            print(f"ask via the Gradio API: {seconds:.1f} s, {len(chat)} messages, textbox cleared: {textbox == ''}")
            print("answer:", final[:400].replace("\n", " | "))
            checks = {
                "answer text": "E3" in final or "overheat" in final.lower(),
                "citation chips": "<kbd>[" in final,
                "passages block": "<details>" in final,
                "grey line with retrieval + cost": "hybrid · cost" in final,
                "status line": "answers left today" in status,
            }
            for name, ok in checks.items():
                print(f"  {'ok ' if ok else 'FAIL'} {name}")
                if not ok:
                    problems.append(name)
            table, chat_after, upload = client.predict(api_name="/clear")[:3]
            print("clear:", "ok" if chat_after == [] else f"chat not reset: {chat_after}")
        except Exception as exc:  # report, do not traceback
            problems.append(f"{type(exc).__name__}: {exc}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    if problems:
        print("SMOKE FAILED:", "; ".join(problems), f"(app log: {args.log})")
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
