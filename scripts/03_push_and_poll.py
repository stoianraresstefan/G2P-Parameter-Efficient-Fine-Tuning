"""Push the GPU kernels to Kaggle, poll until they finish, and pull results.

    python scripts/03_push_and_poll.py --action all    # push -> poll -> pull
    python scripts/03_push_and_poll.py --action push
    python scripts/03_push_and_poll.py --action poll
    python scripts/03_push_and_poll.py --action pull
    python scripts/03_push_and_poll.py --langs rum      # one language only

`push` returns quickly (the run happens server-side). `poll` blocks, checking
status every --interval seconds. `pull` downloads outputs and copies every
per-run JSON into results/runs/ for `build_artifacts.py`.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _util import REPO_ROOT, DATASET_SLUG, get_username, run_kaggle  # noqa: E402

BUILD = REPO_ROOT / "kaggle" / "_build"
RESULTS_RUNS = REPO_ROOT / "results" / "runs"

# Kaggle derives a new kernel's slug from the TITLE, so the metadata `id` must
# already equal that slug or the kernel lands under an unexpected ref (and a
# re-push would create an orphan). Keep id == slugify(title).
KERNEL_TITLE = "G2P PEFT - {lang} (ByT5-G2P LoRA vs full FT)"

# Force a T4 (compute capability 7.5). The current Kaggle image's default GPU
# is not covered by its torch build ("no kernel image is available for execution
# on the device"); T4 is universally supported.
ACCELERATOR = "NvidiaTeslaT4"


def kernel_ref(user: str, lang: str) -> str:
    return f"{user}/g2p-peft-{lang}-byt5-g2p-lora-vs-full-ft"


def build_kernel(user: str, lang: str) -> Path:
    d = BUILD / f"kernel_{lang}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "kaggle" / f"kernel_{lang}" / "run.py", d / "run.py")
    meta = {
        "id": kernel_ref(user, lang),
        "title": KERNEL_TITLE.format(lang=lang),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def push(user, langs):
    for lang in langs:
        d = build_kernel(user, lang)
        run_kaggle(["kernels", "push", "-p", str(d), "--accelerator", ACCELERATOR])
    print(f"Pushed: {', '.join(kernel_ref(user, l) for l in langs)}")


def poll(user, langs, interval=30, timeout=7200) -> bool:
    pending = set(langs)
    start = time.time()
    while pending and (time.time() - start) < timeout:
        for lang in list(pending):
            r = run_kaggle(["kernels", "status", kernel_ref(user, lang)], check=False, capture=True)
            s = (r.stdout + r.stderr).lower()
            # The kaggle CLI (1.6.17+/2.x) prints the KernelWorkerStatus enum repr,
            # e.g. '... has status "KernelWorkerStatus.ERROR"' -> lowercased
            # 'kernelworkerstatus.error'. Parse the quoted token and strip the enum
            # prefix so we match the bare state (also handles older bare-string CLIs).
            m = re.search(r'has status "([\w.]+)"', s)
            if m:
                state = m.group(1).rsplit(".", 1)[-1]
            elif any(x in s for x in ("cannot access", "denied", "not found", "404")):
                # An unparseable status (wrong slug / permissions) — abort this
                # one rather than loop forever misreading it as "running".
                print(f"[{lang}] cannot access kernel — aborting poll: {s.strip()[:90]}")
                pending.discard(lang)
                continue
            else:
                state = s
            if "complete" in state:
                print(f"[{lang}] COMPLETE")
                pending.discard(lang)
            elif "error" in state or "cancel" in state:
                print(f"[{lang}] {state.upper()} — check the kernel log on kaggle.com")
                pending.discard(lang)
            else:
                print(f"[{lang}] running ({state})...")
        if pending:
            time.sleep(interval)
    if pending:
        print(f"WARNING: still pending after timeout: {pending}")
    return not pending


def pull(user, langs):
    RESULTS_RUNS.mkdir(parents=True, exist_ok=True)
    copied = 0
    for lang in langs:
        dst = REPO_ROOT / "results" / f"_kaggle_{lang}"
        dst.mkdir(parents=True, exist_ok=True)
        # --force: overwrite any stale files from a previous version's pull,
        # otherwise `kernels output` silently skips existing filenames.
        run_kaggle(["kernels", "output", kernel_ref(user, lang), "-p", str(dst), "--force"])
        for jf in dst.rglob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict) and "method" in data:
                shutil.copyfile(jf, RESULTS_RUNS / jf.name)
                copied += 1
    print(f"Copied {copied} per-run JSON file(s) into {RESULTS_RUNS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--action", choices=["push", "poll", "pull", "all"], default="all")
    ap.add_argument("--langs", nargs="+", default=["rum", "gre"])
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    user = get_username()
    if a.action in ("push", "all"):
        push(user, a.langs)
    if a.action in ("poll", "all"):
        poll(user, a.langs, interval=a.interval, timeout=a.timeout)
    if a.action in ("pull", "all"):
        pull(user, a.langs)


if __name__ == "__main__":
    main()
