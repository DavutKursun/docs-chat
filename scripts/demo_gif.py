#!/usr/bin/env python
"""Record the demo GIF (docs/img/demo.gif) from the live Space with Playwright.

Playwright is deliberately not in the requirements; use a throwaway venv:

    python3.11 -m venv /tmp/venv-gif
    /tmp/venv-gif/bin/pip install playwright pillow
    /tmp/venv-gif/bin/playwright install chromium
    /tmp/venv-gif/bin/python scripts/demo_gif.py                                   # against the live Space
    /tmp/venv-gif/bin/python scripts/demo_gif.py --url http://127.0.0.1:7860 --out /tmp/demo.gif

Storyboard, under 30 seconds: the page opens with the sample documents already indexed; one question
streams in with [n] citation chips; the cited passage is expanded; a question whose answer is not in the
documents gets the abstention sentence. Frames are taken about twice a second and written with Pillow
(global palette, at most 3 MB). The recording uses two of the demo's daily answers.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

SPACE_URL = "https://davutkursun-docs-chat.hf.space"
ABSTAIN = "I couldn't find this in the uploaded documents."
QUESTION = "What does error code E3 mean and what should I do?"
UNANSWERABLE = "What is the dress code in the office?"


class Recorder:
    """Takes screenshots while the storyboard runs; frames keep their real timestamps."""

    def __init__(self, page: Page, width: int) -> None:
        self.page = page
        self.width = width
        self.frames: list[tuple[Image.Image, float]] = []
        self.t0 = time.perf_counter()

    def snap(self) -> None:
        img = Image.open(io.BytesIO(self.page.screenshot(type="png"))).convert("RGB")
        if img.width != self.width:
            img = img.resize((self.width, round(img.height * self.width / img.width)), Image.LANCZOS)
        self.frames.append((img, time.perf_counter() - self.t0))

    def until(self, ready: Callable[[], bool], timeout: float, interval: float = 0.4) -> bool:
        end = time.perf_counter() + timeout
        while True:
            self.snap()
            if ready():
                return True
            if time.perf_counter() > end:
                return False
            time.sleep(interval)

    def hold(self, seconds: float) -> None:
        self.until(lambda: False, seconds, interval=0.5)

    def type_text(self, text: str, chunk: int = 7) -> None:
        box = self.page.locator("textarea").first
        box.click()
        for i in range(0, len(text), chunk):
            box.press_sequentially(text[i : i + chunk], delay=12)
            self.snap()
        box.press("Enter")


def record(url: str, width: int, headed: bool) -> list[tuple[Image.Image, float]]:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        page = browser.new_page(viewport={"width": 1200, "height": 820}, device_scale_factor=1)
        page.goto(url, wait_until="domcontentloaded")
        rec = Recorder(page, width)

        # 1. the sample documents are indexed when the page opens (no frames while the page loads)
        page.get_by_text("aria_x200_manual.pdf").first.wait_for(timeout=60_000)
        rec.hold(1.5)

        # 2. one question, streamed answer with citation chips and the grey line
        rec.type_text(QUESTION)
        grey_line = page.get_by_text(re.compile(r"\d+\.\d s · \d+ passages"))  # rendered once the stream ends
        if not rec.until(lambda: grey_line.count() > 0, timeout=90, interval=0.3):
            raise SystemExit("no finished answer (grey line) within 90 s")
        if page.locator("kbd").count() == 0:
            raise SystemExit("the answer has no citation chips")
        rec.hold(2.0)

        # 3. expand the cited passage
        summary = page.get_by_text("Show the cited passages").first
        summary.click()
        passage = page.locator("details[open] p").first
        rec.until(lambda: passage.count() > 0, timeout=5)
        summary.scroll_into_view_if_needed()  # chips, summary and the start of the passage stay in view
        rec.hold(3.0)

        # 4. a question whose answer is not in the documents
        rec.type_text(UNANSWERABLE)
        bubble = page.get_by_text(UNANSWERABLE).last  # the chat does not auto-scroll after a manual scroll
        rec.until(lambda: bubble.count() > 0, timeout=10, interval=0.2)
        bubble.scroll_into_view_if_needed()
        abstention = page.get_by_text(ABSTAIN).first
        if not rec.until(lambda: abstention.count() > 0, timeout=90, interval=0.3):
            raise SystemExit("no abstention sentence within 90 s")
        abstention.scroll_into_view_if_needed()
        rec.hold(3.0)
        browser.close()
        return rec.frames


def write_gif(frames: list[tuple[Image.Image, float]], out: Path, max_seconds: float, colors: int) -> tuple[int, float]:
    """Dedupe identical frames, scale durations to fit max_seconds, quantize to one global palette."""
    images: list[Image.Image] = []
    durations: list[float] = []
    for i, (img, t) in enumerate(frames):
        next_t = frames[i + 1][1] if i + 1 < len(frames) else t + 2.0
        d = max(0.05, next_t - t)
        if images and img.tobytes() == images[-1].tobytes():
            durations[-1] += d
        else:
            images.append(img)
            durations.append(d)
    total = sum(durations)
    if total > max_seconds:
        durations = [d * max_seconds / total for d in durations]
    sample = Image.new("RGB", (images[0].width * 3, images[0].height))
    for k, idx in enumerate((0, len(images) // 2, len(images) - 1)):
        sample.paste(images[idx], (k * images[0].width, 0))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quantized = [im.quantize(colors=colors, palette=palette, dither=Image.Dither.NONE) for im in images]
    quantized[0].save(
        out, save_all=True, append_images=quantized[1:], duration=[int(d * 1000) for d in durations],
        loop=0, optimize=True, disposal=1,
    )
    return len(quantized), sum(durations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=SPACE_URL)
    parser.add_argument("--out", type=Path, default=Path("docs/img/demo.gif"))
    parser.add_argument("--width", type=int, default=960, help="GIF width in pixels")
    parser.add_argument("--colors", type=int, default=128)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--max-mb", type=float, default=3)
    parser.add_argument("--headed", action="store_true", help="show the browser while recording")
    args = parser.parse_args(argv)

    frames = record(args.url, args.width, args.headed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    width, colors = args.width, args.colors
    while True:
        n, seconds = write_gif(frames, args.out, args.max_seconds, colors)
        mb = args.out.stat().st_size / 1e6
        print(f"{args.out}: {n} frames, {seconds:.1f} s, {mb:.2f} MB at {width}px / {colors} colours")
        if mb <= args.max_mb:
            return 0
        if colors > 64:
            colors = 64
        elif width > 720:
            width -= 120
            frames = [(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS), t) for im, t in frames]
        else:
            print("could not get under the size limit", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
