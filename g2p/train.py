"""Single-run training/evaluation abstraction.

`run_training(spec, ...)` loads the base model, applies the strategy for `spec`,
trains (unless zero-shot) with dev-set early stopping, evaluates PER/WER on the
test set, evaluates English forgetting PER, and returns a result record dict.

PER/WER are recomputed from decoded predictions (not read from the Trainer's
internal metric) so the reported numbers are guaranteed to follow the
SIGMORPHON convention exactly.
"""

from __future__ import annotations

import shutil
import time
import torch
from pathlib import Path
from transformers import Seq2SeqTrainingArguments, set_seed

from . import config as C
from .data import (
    read_tsv,
    filter_long,
    to_hf_dataset,
    build_input,
    load_english_forgetting,
)
from .metrics import (
    per as per_metric,
    wer as wer_metric,
    make_compute_metrics,
    to_chars,
)
from .model import (
    load_base,
    build_zeroshot,
    build_full_ft,
    build_frozen_head,
    build_lora,
    count_params,
)


def _generate(model, tokenizer, input_texts, device, batch_size: int = 32):
    """Greedy-decode a list of prefixed input strings -> list of phoneme strings."""
    model.eval()
    out_all = []
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=1, max_length=C.GEN_MAX_LENGTH)
        out_all.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out_all


def _evaluate(model, tokenizer, pairs, file_code, device):
    """Return (PER, WER) for (grapheme, gold_tokens) pairs of one language."""
    texts = [build_input(g, file_code) for g, _ in pairs]
    hyps = _generate(model, tokenizer, texts, device)
    scored = [(to_chars(gold), to_chars(h)) for (_, gold), h in zip(pairs, hyps)]
    return per_metric(scored), wer_metric(scored)


def tag_smoke_test(language: str, data_dir, n: int = 50):
    """Sanity check: zero-shot dev PER with the correct CharsiuG2P tag vs. the
    raw SIGMORPHON file code. The correct tag (e.g. `ron`) should score far
    lower than the invalid one (`rum`). Returns {tag: per}.
    """
    model, tokenizer = load_base()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    pairs = filter_long(read_tsv(Path(data_dir) / f"{language}_dev.tsv"))[:n]
    results = {}
    for tag in (C.LANG_TAG[language], language):
        texts = [C.PREFIX_TEMPLATE.format(tag=tag, word=g) for g, _ in pairs]
        hyps = _generate(model, tokenizer, texts, device)
        scored = [(to_chars(gold), to_chars(h)) for (_, gold), h in zip(pairs, hyps)]
        results[tag] = round(per_metric(scored), 2)
    return results


def _build_trainer(model, tokenizer, args, train_ds, dev_ds, patience):
    from transformers import (
        Seq2SeqTrainer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    # `tokenizer` was renamed to `processing_class` in recent transformers.
    try:
        return Seq2SeqTrainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return Seq2SeqTrainer(tokenizer=tokenizer, **kwargs)


def run_training(spec: C.RunSpec, data_dir, eng_test_path, out_dir) -> dict:
    set_seed(spec.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ByT5/T5 are numerically unstable in fp16 (well-documented NaN losses), so
    # prefer bf16 when the GPU supports it and fall back to fp32 otherwise.
    # (T4/P100 lack bf16 -> fp32; ByT5-small at seq-len 128 still fits in 16 GB.)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = False

    model, tokenizer = load_base()

    data_dir = Path(data_dir)
    train_pairs = filter_long(read_tsv(data_dir / f"{spec.language}_train.tsv"))
    dev_pairs = filter_long(read_tsv(data_dir / f"{spec.language}_dev.tsv"))
    test_pairs = filter_long(read_tsv(data_dir / f"{spec.language}_test.tsv"))

    if spec.method == C.ZEROSHOT:
        model = build_zeroshot(model)
    elif spec.method == C.FULL_FT:
        model = build_full_ft(model)
    elif spec.method == C.FROZEN_HEAD:
        model = build_frozen_head(model)
    elif spec.method == C.LORA:
        model = build_lora(model, spec.lora_rank, spec.lora_alpha)
    else:
        raise ValueError(f"unknown method: {spec.method}")

    trainable, total, pct = count_params(model)
    if spec.method == C.ZEROSHOT:
        # Zero-shot trains nothing; the freshly-loaded model has requires_grad=True
        # by default, so report 0 trainable params explicitly.
        trainable, pct = 0, 0.0
    model.to(device)

    train_seconds = 0.0
    ckpt_dir = Path(out_dir) / "ckpt" / spec.name
    if spec.method != C.ZEROSHOT:
        import math

        hp = C.HP[spec.method]
        train_ds = to_hf_dataset(train_pairs, spec.language, tokenizer)
        dev_ds = to_hf_dataset(dev_pairs, spec.language, tokenizer)
        # Evaluate every EVAL_EVERY_EPOCHS epochs (generation eval is the
        # dominant cost). Compute the step interval from the effective batch
        # (accounts for multi-GPU DataParallel on Kaggle's T4 x2).
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
        eff_batch = hp.per_device_train_batch_size * max(1, n_gpu)
        steps_per_epoch = max(1, math.ceil(len(train_pairs) / eff_batch))
        eval_steps = max(1, steps_per_epoch * C.EVAL_EVERY_EPOCHS)
        args = Seq2SeqTrainingArguments(
            output_dir=str(ckpt_dir),
            learning_rate=hp.learning_rate,
            per_device_train_batch_size=hp.per_device_train_batch_size,
            per_device_eval_batch_size=hp.per_device_eval_batch_size,
            num_train_epochs=hp.num_train_epochs,
            warmup_ratio=hp.warmup_ratio,
            weight_decay=hp.weight_decay,
            label_smoothing_factor=hp.label_smoothing_factor,
            predict_with_generate=True,
            generation_num_beams=1,
            generation_max_length=C.GEN_MAX_LENGTH,
            fp16=use_fp16,
            bf16=use_bf16,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            logging_steps=eval_steps,
            report_to=[],
            seed=spec.seed,
        )
        trainer = _build_trainer(
            model, tokenizer, args, train_ds, dev_ds, hp.early_stopping_patience
        )
        t0 = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - t0
        model = trainer.model

    # Final test evaluation, recomputed from decoded predictions.
    test_per, test_wer = _evaluate(model, tokenizer, test_pairs, spec.language, device)

    # Forgetting: English IPA PER for every run (zero-shot gives the baseline).
    eng_pairs = load_english_forgetting(
        eng_test_path, n=C.ENG_FORGETTING_N, seed=C.ENG_FORGETTING_SEED
    )
    forget_per, _ = _evaluate(model, tokenizer, eng_pairs, "eng", device)

    # Persist the tiny LoRA adapter / LM head for reproducibility (full-FT
    # checkpoints are intentionally NOT saved/downloaded).
    _save_artifact(model, spec, out_dir)

    # Free disk: training checkpoints can be large (full FT ~1.2 GB).
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    hp = C.HP.get(spec.method)
    return {
        "language": spec.language,
        "lang_tag": C.LANG_TAG[spec.language],
        "method": spec.method,
        "lora_rank": spec.lora_rank,
        "lora_alpha": spec.lora_alpha,
        "seed": spec.seed,
        "per": round(test_per, 2),
        "wer": round(test_wer, 2),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": round(pct, 4),
        "train_seconds": round(train_seconds, 1),
        "forgetting_per": round(forget_per, 2),
        "base_model": C.BASE_MODEL,
        "epochs": (hp.num_train_epochs if hp else 0),
        "lr": (hp.learning_rate if hp else None),
        "batch_size": (hp.per_device_train_batch_size if hp else None),
        "test_n": len(test_pairs),
        "eng_n": len(eng_pairs),
        "provenance": _provenance(),
    }


def _save_artifact(model, spec: C.RunSpec, out_dir):
    if spec.method == C.LORA:
        adir = Path(out_dir) / "adapters" / spec.name
        adir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adir))
    elif spec.method == C.FROZEN_HEAD:
        import torch

        adir = Path(out_dir) / "adapters" / spec.name
        adir.mkdir(parents=True, exist_ok=True)
        torch.save(model.lm_head.state_dict(), str(adir / "lm_head.pt"))


def _provenance() -> dict:
    info = {}
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import peft

        info["peft"] = peft.__version__
    except Exception:
        pass
    return info
