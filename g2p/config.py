"""Central configuration: model identifiers, language tags, sequence lengths,
per-strategy hyperparameters, and the experiment matrix.

This is the single source of truth for the `rum -> ron` tag correction and the
prefix format. Keeping everything here means the Kaggle kernel and the local
tooling cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

# Model & tokenizer
BASE_MODEL = "charsiu/g2p_multilingual_byT5_small"
TOKENIZER_NAME = "google/byt5-small"
# ByT5 is byte-level
# the model's repo does not provide the tokenizer, so we load it as a
# standalone


# Language tags
# SIGMORPHON file code  ->  CharsiuG2P language tag (ISO 639-3).
# IMPORTANT: Romanian SIGMORPHON files are named `rum_*` (ISO 639-2/B), but
# CharsiuG2P was trained with the ISO 639-3 tag `ron`. Greek matches (`gre`),
# and US English (forgetting set) uses `eng-us`. Using the wrong tag silently
# wrecks the zero-shot baseline.
LANG_TAG = {"rum": "ron", "gre": "gre", "eng": "eng-us"}
# Target adaptation languages (English is used only for the forgetting eval).
LANGUAGES = ("rum", "gre")
# Input format. The space after the colon is REQUIRED by CharsiuG2P.
PREFIX_TEMPLATE = "<{tag}>: {word}"


# Sequence lengths
# The outline filters word pairs longer than 50 CHARACTERS. The model's
# byte-level max_length is set generously because Greek/IPA characters are
# multi-byte (a 50-char Greek word is ~100 UTF-8 bytes), so 50 would truncate.
MAX_CHARS_FILTER = 50
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 128
GEN_MAX_LENGTH = 128


# Method identifiers
ZEROSHOT = "zeroshot"
FROZEN_HEAD = "frozen_head"
LORA = "lora"
FULL_FT = "full_ft"
METHODS = (ZEROSHOT, FROZEN_HEAD, LORA, FULL_FT)


# Params for Eng forgetting eval
ENG_FORGETTING_N = 1000
ENG_FORGETTING_SEED = 42


# Generation-based dev evaluation is the dominant cost, so evaluate every N
# epochs rather than every epoch (early stopping then counts in eval events).
EVAL_EVERY_EPOCHS = 5


@dataclass(frozen=True)
class HParams:
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    label_smoothing_factor: float = 0.1
    early_stopping_patience: int = 3


# Per-strategy defaults for ByT5-small on ~800 training words. LoRA / frozen-head
# tolerate a higher LR than full fine-tuning; label smoothing + warmup + dev
# early stopping guard against the documented low-resource ByT5 instability.
HP = {
    FULL_FT: HParams(learning_rate=5e-4, num_train_epochs=30, weight_decay=0.01),
    LORA: HParams(learning_rate=1e-3, num_train_epochs=40, weight_decay=0.0),
    FROZEN_HEAD: HParams(learning_rate=1e-3, num_train_epochs=40, weight_decay=0.0),
}


SEED = 42


@dataclass(frozen=True)
class RunSpec:
    language: str
    method: str
    lora_rank: int | None = None
    lora_alpha: int | None = None
    seed: int = SEED

    @property
    def name(self) -> str:
        parts = [self.language, self.method]
        if self.method == LORA:
            parts.append(f"r{self.lora_rank}")
        parts.append(f"s{self.seed}")
        return "_".join(parts)


def build_matrix(language: str) -> list[RunSpec]:
    """The runs for one language.

    Base set per language (5 runs): zero-shot, frozen-head, LoRA r=8, LoRA r=16,
    full FT. Romanian additionally gets two extra seeds of LoRA r=8 for the
    headline variance estimate. Total across both languages: 5 + 5 + 2 = 12.
    """
    specs = [
        RunSpec(language, ZEROSHOT, seed=SEED),
        RunSpec(language, FROZEN_HEAD, seed=SEED),
        RunSpec(language, LORA, lora_rank=8, lora_alpha=16, seed=SEED),
        RunSpec(language, LORA, lora_rank=16, lora_alpha=32, seed=SEED),
        RunSpec(language, FULL_FT, seed=SEED),
    ]
    if language == "rum":
        specs += [
            RunSpec(language, LORA, lora_rank=8, lora_alpha=16, seed=1),
            RunSpec(language, LORA, lora_rank=8, lora_alpha=16, seed=2),
        ]
    return specs
