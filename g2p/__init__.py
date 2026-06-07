"""ByT5-G2P parameter-efficient fine-tuning package.

Modules:
  config      central configuration: language tags, paths, hyperparameters, experiment matrix
  data        TSV loading, NFC normalization, prefixed-input construction, HF dataset builder
  metrics     PER / WER replicating the SIGMORPHON 2021 convention
  model       base model + tokenizer loading and the four adaptation strategies
  train       single-run training/evaluation abstraction
  run_matrix  orchestrates the full experiment matrix for one language

The `config`, `data`, and `metrics` modules are importable without torch/transformers
installed (heavy imports are deferred), so data prep and metric tests run locally on CPU.
"""

__all__ = ["config", "data", "metrics", "model", "train", "run_matrix"]
