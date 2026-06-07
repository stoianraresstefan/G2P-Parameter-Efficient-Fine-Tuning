"""Bundle the g2p package + data into a single zip and upload it as a Kaggle
Dataset (so the GPU kernels can import the code and read the data without any
GitHub push).

The dataset contains exactly one file, `g2p_bundle.zip`, plus dataset-metadata.
The kernel unzips it at runtime (kaggle/kernel_*/run.py).

    python scripts/02_make_bundle.py
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _util import REPO_ROOT, BUNDLE_DIR, DATASET_SLUG, get_username, run_kaggle  # noqa: E402

STAGING = REPO_ROOT / "kaggle" / "_staging"
NEEDED = [
    "rum_train.tsv", "rum_dev.tsv", "rum_test.tsv",
    "gre_train.tsv", "gre_dev.tsv", "gre_test.tsv",
    "eng_us_forgetting.tsv",
]


def stage():
    (STAGING / "data").mkdir(parents=True, exist_ok=True)
    missing = [n for n in NEEDED if not (STAGING / "data" / n).exists()]
    if missing:
        raise SystemExit(f"Missing staged data {missing}. Run scripts/01_prepare_data.py first.")
    g2p_dst = STAGING / "g2p"
    if g2p_dst.exists():
        shutil.rmtree(g2p_dst)
    shutil.copytree(
        REPO_ROOT / "g2p", g2p_dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def make_zip() -> Path:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BUNDLE_DIR / "g2p_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(STAGING.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(STAGING).as_posix())
    print(f"  built {zip_path} ({zip_path.stat().st_size // 1024} KiB)")
    return zip_path


def upload(user: str):
    meta = {
        "title": "G2P PEFT bundle (rum/gre code + data)",
        "id": f"{user}/{DATASET_SLUG}",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }
    (BUNDLE_DIR / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    # Decide create-vs-version by checking existence first (the CLI confusingly
    # returns exit 0 even when `create` fails on a title/id collision).
    st = run_kaggle(["datasets", "status", f"{user}/{DATASET_SLUG}"], check=False, capture=True)
    exists = "ready" in (st.stdout + st.stderr).lower()
    if exists:
        run_kaggle(["datasets", "version", "-p", str(BUNDLE_DIR), "-m", "update bundle"])
    else:
        run_kaggle(["datasets", "create", "-p", str(BUNDLE_DIR)])
    print(f"Bundle dataset ready: {meta['id']}")


def main():
    user = get_username()
    stage()
    make_zip()
    upload(user)


if __name__ == "__main__":
    main()
