"""Shared helpers for the local orchestration scripts (Kaggle CLI + paths)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SPLITS = REPO_ROOT / "results" / "data_splits"
BUNDLE_DIR = REPO_ROOT / "kaggle" / "bundle"
DATASET_SLUG = "g2p-peft-bundle"


def kaggle_exe() -> str:
    """Locate the kaggle CLI, preferring the one in the active venv."""
    cand = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if cand.exists():
        return str(cand)
    found = shutil.which("kaggle")
    if found:
        return found
    raise SystemExit(
        "kaggle CLI not found. Install it with:\n"
        '  & ".venv\\Scripts\\pip.exe" install -r requirements-local.txt'
    )


def run_kaggle(args, check=True, capture=False):
    cmd = [kaggle_exe()] + list(args)
    print("+ kaggle " + " ".join(args), flush=True)
    if capture:
        r = subprocess.run(cmd, text=True, capture_output=True)
        if r.stdout:
            print(r.stdout, end="")
        if r.stderr:
            print(r.stderr, end="")
        if check and r.returncode != 0:
            raise SystemExit(f"kaggle command failed: {' '.join(args)}")
        return r
    r = subprocess.run(cmd)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle command failed: {' '.join(args)}")
    return r


def get_username() -> str:
    """Resolve the Kaggle username from env or kaggle.json (never prints the key)."""
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    candidates = []
    cfg = os.environ.get("KAGGLE_CONFIG_DIR")
    if cfg:
        candidates.append(Path(cfg) / "kaggle.json")
    candidates.append(Path.home() / ".kaggle" / "kaggle.json")
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))["username"]
            except Exception:
                pass
    raise SystemExit(
        "Could not determine your Kaggle username.\n"
        "Create an API token at kaggle.com -> Settings -> API -> Create New Token,\n"
        "then save kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json "
        "(or set KAGGLE_USERNAME / KAGGLE_KEY)."
    )


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = os.environ.get("KAGGLE_CONFIG_DIR")
    if cfg and (Path(cfg) / "kaggle.json").exists():
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()
