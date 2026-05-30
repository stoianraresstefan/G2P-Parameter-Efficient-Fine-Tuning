#!/usr/bin/env python
"""Kaggle GPU kernel entrypoint: run the G2P PEFT experiment matrix for one
language. Identical across kernels except for LANG.

Setup (or via scripts/03_push_and_poll.py):
  * Add the `g2p-peft-bundle` dataset as input.
  * Accelerator: GPU (P100 or T4 x2). Internet: ON.

Outputs to /kaggle/working: results_<lang>.json, runs/*.json, smoke_<lang>.json,
and tiny LoRA adapters / LM-head weights under adapters/.
"""
import glob
import os
import subprocess
import sys
import zipfile

LANG = "rum"  # one of: rum, gre

# Diagnostics: show exactly what is mounted under /kaggle/input.
print("=== /kaggle/input tree ===", flush=True)
if os.path.isdir("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count("/") <= 4:
            print(root, "| dirs:", dirs[:12], "| files:", files[:12], flush=True)
else:
    print("/kaggle/input does not exist", flush=True)


def _sh(*args):
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args)


# peft is the only dependency reliably missing/stale in the Kaggle image.
_sh(sys.executable, "-m", "pip", "install", "-q", "peft==0.13.2")

WORK = "/kaggle/working"

# Locate the bundle anywhere under /kaggle/input (recursive — robust to the
# dataset mount path and to whether Kaggle auto-extracted the zip).
pkgs = glob.glob("/kaggle/input/**/g2p/__init__.py", recursive=True)
zips = glob.glob("/kaggle/input/**/g2p_bundle.zip", recursive=True)
if pkgs:
    bundle = os.path.dirname(os.path.dirname(pkgs[0]))
elif zips:
    bundle = os.path.join(WORK, "bundle")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(bundle)
else:
    raise SystemExit(
        "g2p bundle not found under /kaggle/input — is the g2p-peft-bundle "
        "dataset attached to this kernel?"
    )

sys.path.insert(0, bundle)
data_dir = os.path.join(bundle, "data")
eng_test = os.path.join(data_dir, "eng_us_forgetting.tsv")
print("bundle:", bundle, flush=True)
print("data files:", sorted(os.listdir(data_dir)), flush=True)

from g2p.run_matrix import run_language  # noqa: E402

run_language(LANG, data_dir=data_dir, eng_test_path=eng_test, out_dir=WORK)
print("ALL DONE", flush=True)
