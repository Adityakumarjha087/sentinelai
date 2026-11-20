"""Record the README's demo GIF by driving the running dashboard in a headless browser.

Usage, with the stack running and its artifacts built (``docker compose up``)::

    uv run --group demo python scripts/record_demo.py [--ui URL] [--api URL] [--out docs/demo.gif]

Each step of a short tour is captured as a screenshot with a caption bar, and
the screenshots are joined into a looping GIF with Pillow. The browser only
uses the dashboard the way a person would; nothing is changed on the server.
Pass ``--channel chrome`` to use an installed Google Chrome instead of
Playwright's own Chromium (``playwright install chromium``).
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Locator, Page, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT = REPO_ROOT / "agent" / "fonts" / "DejaVuSans.ttf"
BOLD_FONT = REPO_ROOT / "agent" / "fonts" / "DejaVuSans-Bold.ttf"
GIF_WIDTH = 1080
CAPTION_HEIGHT = 64
SLOW = 240_000  # ms; "Investigate all" runs 500 investigations


@dataclass(frozen=True)
class Frame:
    caption: str
    image: Image.Image
    seconds: float


def settle(page: Page, timeout: int = SLOW) -> None:
    """Wait until Streamlit has finished rerunning the script and the page has painted."""
    page.wait_for_timeout(400)
    page.wait_for_selector("[data-testid='stApp'][data-test-script-state='notRunning']", timeout=timeout)
    page.wait_for_timeout(1200)


_REVEAL = """(el, margin) => {
    el.scrollIntoView({block: "start"});
    let box = el.parentElement;
    while (box && !["auto", "scroll"].includes(getComputedStyle(box).overflowY)) box = box.parentElement;
    (box || document.scrollingElement).scrollBy(0, -margin);
}"""


def reveal(locator: Locator, margin: int = 90) -> None:
    """Scroll so ``locator`` sits near the top of the screen, below Streamlit's header."""
    locator.first.evaluate(_REVEAL, margin)
    locator.page.wait_for_timeout(500)


def pick(page: Page, key: str, text: str) -> None:
    """Choose the first selectbox option containing ``text``."""
    page.locator(f".st-key-{key} input").click()
    page.keyboard.type(text)
    page.locator("[role='option']", has_text=text).first.click()
    settle(page)


def captioned(screenshot: bytes, step: int, total: int, caption: str) -> Image.Image:
    """The screenshot under a dark caption bar, scaled to the GIF width."""
    shot = Image.open(io.BytesIO(screenshot)).convert("RGB")
    canvas = Image.new("RGB", (shot.width, shot.height + CAPTION_HEIGHT), (17, 24, 39))
    canvas.paste(shot, (0, CAPTION_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    badge = f"{step}/{total}"
    draw.text((24, 18), badge, font=ImageFont.truetype(str(BOLD_FONT), 24), fill=(129, 140, 248))
    draw.text((96, 18), caption, font=ImageFont.truetype(str(FONT), 24), fill=(243, 244, 246))
    height = round(canvas.height * GIF_WIDTH / canvas.width)
    return canvas.resize((GIF_WIDTH, height), Image.Resampling.LANCZOS)


def tour(page: Page, ui: str, api: str | None) -> list[tuple[str, float, Callable[[], None]]]:
    """The steps: (caption, seconds on screen, action that sets the scene)."""

    def open_host() -> None:
        page.goto(ui, wait_until="networkidle")
        page.wait_for_selector(".st-key-host-one", timeout=SLOW)
        settle(page)

    def investigate_injected() -> None:
        pick(page, "host-pick", "2d6d6719")
        page.locator(".st-key-host-one button").click()
        settle(page)

    def open_report() -> None:
        page.locator("[data-testid='stExpander'] summary", has_text="Incident report").first.click()
        settle(page)
        reveal(page.locator("[data-testid='stExpander']", has_text="Incident report"))

    def open_network() -> None:
        page.evaluate("window.scrollTo(0, 0)")
        page.locator("[data-testid='stTab']", has_text="Network detections").click()
        page.wait_for_selector(".st-key-flow-one", timeout=SLOW)
        settle(page)

    def investigate_all_flows() -> None:
        page.locator(".st-key-flow-all button").click()
        settle(page)
        reveal(page.get_by_text("Severity of investigated alerts").last, margin=160)

    def investigate_scan() -> None:
        pick(page, "flow-pick", "172.16.0.1 ->")
        page.locator(".st-key-flow-one button").click()
        settle(page)
        page.locator("[data-testid='stExpander'] summary", has_text="What the analyst sees").last.click()
        settle(page)
        reveal(page.locator("[data-testid='stExpander']", has_text="What the analyst sees").last)

    steps: list[tuple[str, float, Callable[[], None]]] = [
        ("Host events: a Sysmon sample. One command line hides a prompt injection.", 3.5, open_host),
        ("Investigated: encoded PowerShell, T1059.001. The injection is flagged, not obeyed.", 4.5,
         investigate_injected),
        ("Incident report: ATT&CK mapping and mitigations from the knowledge base.", 4.0, open_report),
        ("Network: the autoencoder flagged 162,678 of 286,467 port-scan-day flows.", 4.0, open_network),
        ("500 flagged flows investigated by the offline baseline analyst.", 3.5, investigate_all_flows),
        ("What the analyst sees: behaviour and score, no dataset label.", 4.5, investigate_scan),
    ]  # fmt: skip
    if api is not None:

        def open_api() -> None:
            page.goto(f"{api}/docs", wait_until="networkidle")
            page.wait_for_selector("text=/detections", timeout=SLOW)
            page.wait_for_timeout(800)

        steps.append(("The same investigations over a REST API (FastAPI).", 3.0, open_api))
    return steps


def record(ui: str, api: str | None, out: Path, channel: str | None) -> list[Frame]:
    frames: list[Frame] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        steps = tour(page, ui, api)
        for number, (caption, seconds, action) in enumerate(steps, start=1):
            action()
            frames.append(Frame(caption, captioned(page.screenshot(), number, len(steps), caption), seconds))
            print(f"captured {number}/{len(steps)}: {caption}")
        browser.close()
    first, *rest = (frame.image for frame in frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    first.save(
        out,
        save_all=True,
        append_images=rest,
        duration=[round(frame.seconds * 1000) for frame in frames],
        loop=0,
        optimize=True,
    )
    return frames


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record the dashboard demo GIF.")
    parser.add_argument("--ui", default="http://127.0.0.1:8501")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API URL for the last frame; 'none' to skip")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "demo.gif")
    parser.add_argument("--channel", default=None, help="browser channel, e.g. 'chrome'")
    args = parser.parse_args(argv)
    frames = record(args.ui.rstrip("/"), None if args.api == "none" else args.api.rstrip("/"), args.out, args.channel)
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({len(frames)} frames, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
