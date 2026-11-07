"""Download Poly-X model weights into the user data directory.

Usage:
    polyx-download-models
    polyx-download-models --model bestdetectormedium.pt
    polyx-download-models --url https://example.com/model.pt --name model.pt
    polyx-download-models --sha256 <expected-digest>

Model weights (.pt) are PyTorch pickles: running one executes code. For any
untrusted source, pass --sha256 with the digest published alongside the
release; the download aborts on mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from polyx.core.i18n import _
from polyx.core.paths import MODELS_DIR

DEFAULT_URL = os.environ.get(
    "POLYX_MODEL_URL",
    "https://github.com/CrissFerrada/Poly-X-Microplastics/releases/latest/download/bestdetectormedium.pt",
)
DEFAULT_NAME = "bestdetectormedium.pt"
CHUNK = 1024 * 64


def _progress(done: int, total: int) -> None:
    if total > 0:
        pct = min(100, int(done * 100 / total))
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        sys.stdout.write(
            _("\r[%(bar)s] %(pct)3d%%  %(done)6.1f/%(total)6.1f MB")
            % {"bar": bar, "pct": pct, "done": done / 1e6, "total": total / 1e6}
        )
    else:
        sys.stdout.write(_("\rDownloaded %(done)6.1f MB") % {"done": done / 1e6})
    sys.stdout.flush()


def download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download url to dest, optionally verifying sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = Request(url, headers={"User-Agent": "Poly-X"})  # noqa: S310
        with urlopen(req) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length", 0) or 0)
            done = 0
            h = hashlib.sha256()
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    _progress(done, total)
            print()
        if expected_sha256 and h.hexdigest().lower() != expected_sha256.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                _("SHA-256 mismatch: got %(got)s, expected %(expected)s")
                % {"got": h.hexdigest(), "expected": expected_sha256}
            )
        tmp.replace(dest)
    except (URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(_("Download failed: %(error)s") % {"error": exc}) from exc
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=_("Download Poly-X model weights"))
    parser.add_argument("--model", default=DEFAULT_NAME, help=_("output filename"))
    parser.add_argument("--url", default=DEFAULT_URL, help=_("download URL"))
    parser.add_argument("--sha256", default=None, help=_("expected SHA-256 (optional)"))
    parser.add_argument("--force", action="store_true", help=_("overwrite existing"))
    args = parser.parse_args(argv)

    dest = MODELS_DIR / args.model
    if dest.exists() and not args.force:
        print(_("Already present: %(path)s") % {"path": dest})
        return 0

    print(_("Downloading %(url)s -> %(path)s") % {"url": args.url, "path": dest})
    try:
        download(args.url, dest, expected_sha256=args.sha256)
    except RuntimeError as exc:
        print(_("ERROR: %(error)s") % {"error": exc}, file=sys.stderr)
        return 1
    print(_("Saved: %(path)s") % {"path": dest})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
