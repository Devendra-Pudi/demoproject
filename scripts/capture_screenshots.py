"""Capture real screenshots of every app for the READMEs and the Pages directory.

Unlike ``scripts/make_previews.py`` (which renders structure from the running code), this script
drives a real browser through Playwright and writes pixel-accurate PNGs. It needs a browser that
may not be available in a restricted environment, so it skips cleanly instead of failing.

    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py            # → docs/screenshots/*.png

Each app is started on its own port, screenshotted, and stopped again. Nothing is uploaded.
"""
from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
PYTHON = sys.executable

APPS = [
    ("production-rag-application", "01-production-rag",
     [PYTHON, "-m", "streamlit", "run", "projects/production-rag-application/app.py",
      "--server.port", "8501", "--server.address", "127.0.0.1", "--server.headless", "true"],
     "http://127.0.0.1:8501"),
    ("local-slm-ollama", "02-local-slm",
     [PYTHON, "-m", "streamlit", "run", "projects/local-slm-ollama/app.py",
      "--server.port", "8502", "--server.address", "127.0.0.1", "--server.headless", "true"],
     "http://127.0.0.1:8502"),
    ("monitoring-observability", "03-monitoring",
     [PYTHON, "-m", "streamlit", "run", "projects/monitoring-observability/app.py",
      "--server.port", "8503", "--server.address", "127.0.0.1", "--server.headless", "true"],
     "http://127.0.0.1:8503"),
    ("lora-dpo-fine-tuning", "04-fine-tuning",
     [PYTHON, "-m", "streamlit", "run", "projects/lora-dpo-fine-tuning/app.py",
      "--server.port", "8504", "--server.address", "127.0.0.1", "--server.headless", "true"],
     "http://127.0.0.1:8504"),
]

BUTTONS = {
    "02-local-slm": "Load the illustrative example layout",
    "03-monitoring": "Load 24 synthetic example traces",
    "04-fine-tuning": "Load the illustrative example layout",
}


def wait_for(port: int, timeout: float = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.closing(socket.socket()) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def capture(page, url: str, output: Path, button: str | None):
    page.goto(url, wait_until="networkidle", timeout=60_000)
    if button:
        with contextlib.suppress(Exception):
            page.get_by_role("button", name=button).click(timeout=15_000)
            page.wait_for_timeout(2500)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(output), full_page=True)
    print("captured", output.relative_to(ROOT))


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed; run: pip install playwright && playwright install chromium")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as error:  # browser binaries missing in restricted environments
            print(f"No browser available ({error.__class__.__name__}); nothing captured.")
            return 0
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        for _folder, name, command, url in APPS:
            server = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,
                                      env={**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"})
            try:
                if not wait_for(int(url.rsplit(":", 1)[1])):
                    print(f"{name}: app did not start; skipped")
                    continue
                capture(page, url, OUT / f"{name}.png", BUTTONS.get(name))
            finally:
                server.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    server.wait(timeout=15)
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
