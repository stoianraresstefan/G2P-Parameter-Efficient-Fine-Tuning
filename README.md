# G2P Parameter-Efficient Fine-Tuning

Code for the study *LoRA for Multilingual Grapheme-to-Phoneme Conversion:
Parameter-Efficient Adaptation of ByT5-G2P on Romanian and Modern Greek*.

We benchmark four adaptation strategies for the `charsiu/g2p_multilingual_byT5_small`
model on the SIGMORPHON 2021 low-resource Romanian (`rum`) and Greek (`gre`)
splits, with a catastrophic-forgetting check on English (eng-us) IPA:

| Strategy        | What trains                          |
| --------------- | ------------------------------------ |
| Zero-shot       | nothing (pretrained baseline)        |
| Frozen head     | only the LM head                     |
| LoRA (r=8, r=16)| rank-r adapters on attention q/k/v/o |
| Full FT         | all ~300M parameters                 |

GPU training runs on **Kaggle**; everything else (data prep, metrics, plots,
orchestration) runs locally on CPU.

## Layout

```
g2p/         ML core (runs on Kaggle; data/metrics importable locally)
analysis/    results -> LaTeX tables + vector PDF figures (CPU only)
kaggle/      kernel scripts, dataset bundle, manual fallback notebook
scripts/     local driver: setup, data download, bundle upload, push/poll/pull
tests/       PER/WER + Pareto unit tests
build_artifacts.py   regenerate all tables/figures from results/runs/*.json
```

## Workflow

```powershell
# 0. one-time setup: install local deps + check Kaggle token
.\scripts\00_setup.ps1
#    (create the token at kaggle.com -> Settings -> API; save kaggle.json to
#     %USERPROFILE%\.kaggle\kaggle.json)

# 1. download SIGMORPHON data + build the English forgetting sample
python scripts\01_prepare_data.py

# 2. bundle g2p/ + data as a Kaggle dataset
python scripts\02_make_bundle.py

# 3. push the two GPU kernels, wait, and pull results into results/runs/
python scripts\03_push_and_poll.py

# 4. regenerate Table I/II (tables/) and Figures 1/2 (figures/)
python build_artifacts.py
```

Dry-run the analysis pipeline without any GPU:

```powershell
python scripts\make_demo_results.py   # writes synthetic results/runs/*.json
python build_artifacts.py
python -m pytest tests\                # or: python tests\test_metrics.py
python analysis\overlap_check.py       # train/test overlap validity report
```

## Notes

* **Language tags:** CharsiuG2P uses `<ron>` for Romanian (the SIGMORPHON files
  are named `rum_*`), `<gre>` for Greek, `<eng-us>` for English. Centralized in
  `g2p/config.py`.
* **Forgetting metric:** measured against English **IPA** (SIGMORPHON eng-us),
  not CMUDict ARPAbet, so predictions and references share the same phoneme
  alphabet.
* **GPU:** a single 16 GB GPU (P100 or T4) is sufficient for ByT5-small. T4 x2 is
  only reachable via the manual notebook (`kaggle/fallback_notebook.ipynb`).
* Credentials live in `%USERPROFILE%\.kaggle\kaggle.json` (outside the repo) and
  are never committed.
