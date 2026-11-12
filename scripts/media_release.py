#!/usr/bin/env python3
"""Check, stamp, and upload versioned README .webp animations to GitHub Releases."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MANIFEST = DOCS / "manifest.json"
REPO = "ryanchaiyakul/dismech-newton"

# Map each README .webp asset to the example script that generates it.
ASSETS: dict[str, str] = {
    "cantilever.webp": "examples/cantilever.py",
    "knot.webp": "examples/overhand_knot.py",
    "plectoneme.webp": "examples/plectoneme.py",
    "flagella.webp": "examples/flagella.py",
    "shell_cantilever.webp": "examples/shell_cantilever.py",
    "shell_fold.webp": "examples/shell_fold.py",
    "cloth_twist.webp": "examples/newton/cloth_twist.py",
    "franka.webp": "examples/newton/franka_cable_ik_pick_place.py",
}


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find version in pyproject.toml")
    return m.group(1)


def _ast_digest(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def _shared_hash() -> str:
    """Hash ASTs of solver and shared example utilities."""
    h = hashlib.sha256()
    files = sorted((ROOT / "src" / "dismech_newton").glob("*.py"))
    files += sorted((ROOT / "examples" / "utils").glob("*.py"))
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode("utf-8"))
        h.update(_ast_digest(p).encode("utf-8"))
    return h.hexdigest()


def asset_hashes() -> dict[str, str]:
    shared = _shared_hash()
    out: dict[str, str] = {}
    for webp, script in ASSETS.items():
        h = hashlib.sha256()
        h.update(shared.encode("utf-8"))
        h.update(_ast_digest(ROOT / script).encode("utf-8"))
        out[webp] = h.hexdigest()[:16]
    return out


def cmd_stamp() -> int:
    DOCS.mkdir(exist_ok=True)
    hashes = asset_hashes()
    missing = [w for w in ASSETS if not (DOCS / w).is_file()]
    if missing:
        print(f"Missing local .webp files in docs/: {', '.join(missing)}", file=sys.stderr)
        return 1
    MANIFEST.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Stamped {len(hashes)} assets in {MANIFEST.relative_to(ROOT)}")
    return 0


def cmd_check() -> int:
    version = project_version()
    tag = f"v{version}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    errors: list[str] = []

    for webp in ASSETS:
        expected_url = f"https://github.com/{REPO}/releases/download/{tag}/{webp}"
        if expected_url not in readme:
            errors.append(f"README.md missing release URL for {tag}: {expected_url}")
        if not (DOCS / webp).is_file():
            errors.append(f"Missing cached file: docs/{webp}")

    current = asset_hashes()
    recorded: dict[str, str] = {}
    if MANIFEST.is_file():
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        errors.append("Missing docs/manifest.json (run `python3 scripts/media_release.py stamp` after updating .webps)")

    for webp, digest in current.items():
        if recorded and recorded.get(webp) != digest:
            errors.append(f"Stale docs/{webp} ({ASSETS[webp]} or src/ changed); re-render and run `stamp`")

    if errors:
        for err in errors:
            print(f"media-check: {err}", file=sys.stderr)
        return 1
    print(f"media-check: all {len(ASSETS)} .webp assets up to date for {tag}")
    return 0


def cmd_upload() -> int:
    version = project_version()
    tag = f"v{version}"
    files = [str(DOCS / w) for w in ASSETS]
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        print(f"Missing local .webp files: {', '.join(missing)}", file=sys.stderr)
        return 1

    gh = shutil.which("gh")
    if not gh:
        print(
            f"gh CLI not found. Upload docs/*.webp to GitHub Release {tag} manually "
            f"(or set SKIP_MEDIA=1 to bypass).",
            file=sys.stderr,
        )
        return 1

    view = subprocess.run([gh, "release", "view", tag], cwd=ROOT, capture_output=True)
    if view.returncode != 0:
        subprocess.run(
            [gh, "release", "create", tag, "--title", tag, "--notes", f"Release {tag} media assets"],
            cwd=ROOT,
            check=True,
        )
    subprocess.run([gh, "release", "upload", tag, "--clobber", *files], cwd=ROOT, check=True)
    print(f"Uploaded {len(files)} .webp assets to GitHub Release {tag}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        return cmd_check()
    if cmd == "stamp":
        return cmd_stamp()
    if cmd == "upload":
        return cmd_upload()
    print(f"Usage: {sys.argv[0]} {{check|stamp|upload}}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
