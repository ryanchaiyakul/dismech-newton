"""Warp kernel cache of pushed commits for Colab's T4, so a test run there loads its kernels instead of compiling them
(cold 520 s, restored 70 s on 2026-10-27). Stdlib only: it runs on a fresh Colab VM before ``uv sync``.

    python scripts/warp_cache.py build SHA      # .githooks/pre-push runs this in the background per pushed commit
    python scripts/warp_cache.py restore [REF] [DIR]

``build`` checks SHA out in ``.cache/warp-cache/tree``, runs the tests there into an empty kernel cache (as T4 PTX,
plus every CUDA module as sm_75 CUBIN: ``--cubin-arch``) and uploads it as ``SHA-sm75.tar.gz`` to the
``warp-cache`` release (``gh``), keeping the newest ``KEEP``. ``restore`` unpacks the cache of REF (default HEAD) or
its nearest cached ancestor into DIR (default ``$WARP_CACHE_PATH``, else Warp's): then ``pytest --cubin`` compiles
only the modules changed since. Warp's CPU modules (``test_devices_agree``) do not carry over from Windows.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ARCH = 75
RELEASE = "warp-cache"
KEEP = 30
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".cache" / "warp-cache"


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def asset(sha):
    return f"{sha}-sm{ARCH}.tar.gz"


def repo():
    """``owner/name`` of origin, from either remote URL form."""
    return re.search(r"github\.com[:/](.+?)(?:\.git)?$", git("remote", "get-url", "origin")).group(1)


def released():
    """Asset name -> download URL of the release (public, no token)."""
    url = f"https://api.github.com/repos/{repo()}/releases/tags/{RELEASE}"
    try:
        with urllib.request.urlopen(url) as r:
            return {a["name"]: a["browser_download_url"] for a in json.load(r)["assets"]}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}
        raise


def gh(*args, check=True):
    return subprocess.run(["gh", *args, "-R", repo()], cwd=ROOT, check=check, capture_output=True, text=True)


def lock():
    """One build at a time (they share the GPU and the worktree); a lock over an hour old is a dead build's."""
    path = WORK / "lock"
    while True:
        try:
            path.mkdir()
            return path
        except FileExistsError:
            if time.time() - path.stat().st_mtime > 3600:
                path.rmdir()
            time.sleep(10)


def build(sha):
    sha = git("rev-parse", sha)
    if asset(sha) in released():
        print(f"{asset(sha)} already released")
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    held = lock()
    try:
        tree, cache = WORK / "tree", WORK / f"build-{sha}"
        if tree.exists():
            git("checkout", "--detach", "--force", sha, cwd=tree)
            git("clean", "-fdx", "-e", ".venv", cwd=tree)
        else:
            git("worktree", "add", "--detach", str(tree), sha)
        shutil.rmtree(cache, ignore_errors=True)
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"} | {"WARP_CACHE_PATH": str(cache)}
        t = time.time()
        tests = subprocess.run(["uv", "run", "--extra", "gpu", "--group", "dev", "pytest", "-q", "-p", "no:cacheprovider",
                                "--ptx-arch", str(ARCH), "--cubin-arch", str(ARCH)], cwd=tree, env=env)
        print(f"tests exit {tests.returncode} in {time.time() - t:.0f} s")
        if not any(cache.glob(f"*/wp_*/*.sm{ARCH}.cubin")):
            print("no CUBIN built: nothing to upload")
            return 1
        tar = WORK / asset(sha)
        with tarfile.open(tar, "w:gz") as f:
            for version in cache.iterdir():
                f.add(version, arcname=version.name)
        if gh("release", "view", RELEASE, check=False).returncode:
            gh("release", "create", RELEASE, "--prerelease", "--title", "Warp kernel cache",
               "--notes", "T4 (sm_75) Warp kernel caches of pushed commits: scripts/warp_cache.py. Not a release.")
        gh("release", "upload", RELEASE, str(tar), "--clobber")
        print(f"uploaded {tar.name} ({tar.stat().st_size / 1e6:.1f} MB)")
        assets = json.loads(gh("release", "view", RELEASE, "--json", "assets").stdout)["assets"]
        for a in sorted(assets, key=lambda a: a["createdAt"])[:-KEEP]:
            gh("release", "delete-asset", RELEASE, a["name"], "--yes")
        tar.unlink()
        shutil.rmtree(cache, ignore_errors=True)
        return tests.returncode
    finally:
        held.rmdir()


def restore(ref="HEAD", dest=None):
    if dest is None:
        dest = os.environ.get("WARP_CACHE_PATH")
    if dest is None:
        import warp as wp

        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        dest = os.path.dirname(wp.config.kernel_cache_dir)
    urls = released()
    for n, sha in enumerate(git("rev-list", "--max-count", "100", ref).split()):
        if asset(sha) in urls:
            print(f"restoring {asset(sha)} ({n} commits back) into {dest}")
            tar = Path(dest) / asset(sha)
            tar.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(urls[asset(sha)], tar)
            with tarfile.open(tar) as f:
                f.extractall(dest, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
            tar.unlink()
            return 0
    print(f"no cached commit among {ref}'s last 100 (shallow clone? fetch with --filter=blob:none)")
    return 1


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    sys.exit({"build": build, "restore": restore}[command](*args))
