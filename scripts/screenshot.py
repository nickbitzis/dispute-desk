"""Capture README screenshots with headless Chrome. Start the server first: make run.

Usage: uv run python scripts/screenshot.py --port 8833
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
]
DOCS = Path(__file__).resolve().parents[1] / "docs"


def chrome() -> str:
    for c in CHROME_PATHS:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Chrome not found")


def shoot(url: str, out: Path, width: int, height: int) -> None:
    args = [
        chrome(),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--virtual-time-budget=8000",
        f"--screenshot={out}",
        url,
    ]
    subprocess.run(args, check=True, capture_output=True, timeout=60)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8833)
    ap.add_argument("--dispute", default=None, help="dispute id for the detail page; defaults to a fraud dispute")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    did = a.dispute
    if did is None:
        with urllib.request.urlopen(f"{base}/api/queue?limit=40&category=fraud") as r:
            items = json.load(r)["items"]
        did = next((i["dispute_id"] for i in items if i["recommendation"] == "represent"), items[0]["dispute_id"])
    DOCS.mkdir(exist_ok=True)
    shoot(f"{base}/#/queue", DOCS / "screenshot.png", 1400, 1000)
    shoot(f"{base}/#/dispute/{did}", DOCS / "dispute-detail.png", 1400, 1500)
    shoot(f"{base}/#/analytics", DOCS / "analytics.png", 1400, 1650)
    shoot(f"{base}/#/policy", DOCS / "policy.png", 1400, 1100)


if __name__ == "__main__":
    main()
