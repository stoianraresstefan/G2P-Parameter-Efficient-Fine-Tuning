"""Model + tokenizer loading and the four adaptation strategies.

Heavy imports (transformers, peft, torch) are deferred to call time so the rest
of the package stays importable on a CPU-only machine without these installed.
"""

from __future__ import annotations

from .config import BASE_MODEL, TOKENIZER_NAME

from transformers import T5ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_base():
    """Load the CharsiuG2P ByT5 model and the ByT5 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
    return model, tokenizer


def build_zeroshot(model):
    """No-op: the unadapted base model used for inference only."""
    model.eval()
    return model


def build_full_ft(model):
    """All parameters trainable."""
    for p in model.parameters():
        p.requires_grad = True
    return model


def build_frozen_head(model):
    """Freeze the entire base; train only the LM head.

    Valid because the CharsiuG2P checkpoint has `tie_word_embeddings=False`, so
    `lm_head` is an independent Linear (not tied to the input embeddings).
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True
    return model


def build_lora(model, r: int, alpha: int, dropout: float = 0.0):
    """Inject LoRA into the attention q/k/v/o projections.

    PEFT suffix-matches `target_modules`, so ["q","k","v","o"] patches all four
    projections across self- and cross-attention in both encoder and decoder.
    """

    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=["q", "k", "v", "o"],
        lora_dropout=dropout,
        bias="none",
    )
    return get_peft_model(model, cfg)


def count_params(model):
    """Return (trainable, total, trainable_pct) for the (possibly wrapped) model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct
