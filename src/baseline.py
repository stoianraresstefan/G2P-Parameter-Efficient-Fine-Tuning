"""
Baseline: Zero-Shot ByT5 G2P
Model : charsiu/g2p_multilingual_byT5_small
Task  : SIGMORPHON 2021 Shared Task 1 — Subtask 3
Langs : Romanian (rum), Modern Greek (gre)

No training. Greedy decoding. Language-code prefix per CharsiuG2P convention.
Establishes the "no adaptation" lower bound before LoRA fine-tuning.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "charsiu/g2p_multilingual_byT5_small"
TOKENIZER_ID = "google/byt5-small"  # CharsiuG2P has no tokenizer of its own

# CharsiuG2P uses its own ISO-639-based codes (see lang_list.txt in their repo).
# - Modern Greek : <gre>  confirmed in lang_list.txt
# - Romanian     : <ron>  is NOT in lang_list.txt — model was not trained on it.
#                  We use <ron> as a best-effort probe; this is a true zero-shot.
LANG_PREFIX = {
    "rum": "<ron>",  # Romanian ISO-639-2 — NOT in CharsiuG2P training languages
    "gre": "<gre>",  # Modern Greek       — confirmed in CharsiuG2P training languages
}
LANG_SEEN_BY_MODEL = {
    "rum": False,  # true zero-shot: unseen language
    "gre": True,  # cross-lingual zero-shot: seen language, unseen SIGMORPHON words
}

TARGET_LANGS = list(LANG_PREFIX.keys())


# ── Data ─────────────────────────────────────────────────────────────────────


def load_split(path: Path) -> list[tuple[str, str | None]]:
    """
    Load a SIGMORPHON TSV split.
    Returns list of (grapheme_seq, phoneme_seq_or_None).
    Test files have only one column; dev/train have two.
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            grapheme = parts[0].strip()
            phoneme = parts[1].strip() if len(parts) > 1 else None
            examples.append((grapheme, phoneme))
    return examples


# ── Inference ────────────────────────────────────────────────────────────────


def build_input(grapheme: str, lang: str) -> str:
    """
    Format input as CharsiuG2P expects:
        <ron>: antonim
    SIGMORPHON stores graphemes as space-separated chars ('a n t o n i m'),
    so we join them back into a plain word first.
    The space after the colon is mandatory per CharsiuG2P docs.
    """
    word = grapheme.replace(" ", "")  # 'a n t o n i m' -> 'antonim'
    prefix = LANG_PREFIX[lang]
    return f"{prefix}: {word}"


def predict_greedy(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    graphemes: list[str],
    lang: str,
    batch_size: int = 64,
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> list[str]:
    """
    Greedy decoding. CharsiuG2P authors found beam search gives no improvement
    on this model, so we use greedy (num_beams=1) for speed and simplicity.
    """
    predictions = []

    for i in range(0, len(graphemes), batch_size):
        batch_graphemes = graphemes[i : i + batch_size]
        inputs_text = [build_input(g, lang) for g in batch_graphemes]

        enc = tokenizer(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=False,  # required by CharsiuG2P
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=1,  # greedy
                do_sample=False,
            )

        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        predictions.extend(tokenize_ipa(p) for p in decoded)

        done = min(i + batch_size, len(graphemes))
        print(f"    {done}/{len(graphemes)}", end="\r")

    print()
    return predictions


# ── IPA tokenization ─────────────────────────────────────────────────────────

# Multi-character IPA phones that must be kept together, ordered longest-first
# so the greedy match always picks the longest valid token.
_IPA_MULTICHAR = sorted(
    [
        # Affricates with tie bar (U+0361 or U+035C)
        "t͡ʃ",
        "d͡ʒ",  # t͡ʃ  d͡ʒ
        "t͜ʃ",
        "d͜ʒ",
        "t͡s",
        "d͡z",  # t͡s  d͡z
        "t͜s",
        "d͜z",
        "t͡ɬ",  # t͡ɬ
        # Common diacritics that follow a base (keep base+diacritic together)
        # Handled by including length-2+ combos in the sorted list
    ],
    key=len,
    reverse=True,
)

# Unicode combining diacritics that attach to the preceding character
_COMBINING = set(range(0x0300, 0x036F + 1)) | {
    0x02B0,
    0x02B1,
    0x02B2,
    0x02B7,
    0x02B8,  # modifier letters
    0x02BC,
    0x02C0,
    0x02C1,
    0x0325,
    0x032A,
    0x033A,
    0x033B,
    0x033C,
    0x207F,  # superscript n
}


def tokenize_ipa(ipa_string: str) -> str:
    """
    Convert a raw CharsiuG2P output string into a space-separated phone sequence
    that matches SIGMORPHON's format.

    Strategy: greedy longest-match over known multi-char phones, then
    character-by-character while attaching combining diacritics to their base.

    Examples:
        'abandonat'   -> 'a b a n d o n a t'
        'aktʃeptat'   -> 'a k t ʃ e p t a t'   (ʃ is single char here)
        'at͡ʃe'        -> 'a t͡ʃ e'              (tie-bar affricate kept together)
        'anʲ'         -> 'a nʲ'                 (modifier letter attached)
    """
    s = ipa_string.strip()
    phones = []
    i = 0
    while i < len(s):
        # Skip spaces already in the string
        if s[i] == " ":
            i += 1
            continue

        # Try longest multi-char phone first
        matched = False
        for mc in _IPA_MULTICHAR:
            if s[i : i + len(mc)] == mc:
                phones.append(mc)
                i += len(mc)
                matched = True
                break

        if not matched:
            # Take one character as base
            phone = s[i]
            i += 1
            # Attach any immediately following combining diacritics / modifiers
            while i < len(s) and ord(s[i]) in _COMBINING:
                phone += s[i]
                i += 1
            phones.append(phone)

    return " ".join(phones)


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_wer(gold: list[str], hyp: list[str]) -> float:
    """
    Word Error Rate as defined by SIGMORPHON:
    percentage of words where predicted phone sequence ≠ gold (exact match).
    Returned as a value in [0, 100].
    """
    assert len(gold) == len(hyp)
    errors = sum(1 for g, h in zip(gold, hyp) if g != h)
    return 100.0 * errors / len(gold)


def compute_per(gold: list[str], hyp: list[str]) -> float:
    """
    Phone Error Rate: character-level edit distance averaged over sequences.
    Useful diagnostic alongside WER for the low-resource setting.
    """

    def edit_distance(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = (
                    prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
                )
                prev = temp
        return dp[n]

    total_dist, total_phones = 0, 0
    for g, h in zip(gold, hyp):
        g_phones = g.split()
        h_phones = h.split()
        total_dist += edit_distance(g_phones, h_phones)
        total_phones += max(len(g_phones), 1)

    return 100.0 * total_dist / total_phones


# ── Output ───────────────────────────────────────────────────────────────────


def write_hypotheses(path: Path, graphemes: list[str], preds: list[str]):
    """Write two-column TSV: grapheme \\t predicted_phones"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for g, p in zip(graphemes, preds):
            f.write(f"{g}\t{p}\n")


def print_samples(graphemes, gold, preds, n=8):
    print(f"    {'GRAPHEME':<25} {'GOLD':<30} {'PRED'}")
    print(f"    {'-' * 25} {'-' * 30} {'-' * 30}")
    for g, go, pr in zip(graphemes[:n], gold[:n], preds[:n]):
        match = "✓" if go == pr else "✗"
        print(f" {match}  {g:<25} {go:<30} {pr}")


# ── Main ─────────────────────────────────────────────────────────────────────


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 60}")
    print(f"Baseline: Zero-Shot ByT5 G2P ({MODEL_ID})")
    print(f"Device  : {device}")
    print(f"{'=' * 60}\n")

    print(f"Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    all_results = {}

    for lang in TARGET_LANGS:
        print(f"{'─' * 60}")
        seen = (
            "seen by model"
            if LANG_SEEN_BY_MODEL[lang]
            else "UNSEEN by model (true zero-shot)"
        )
        print(f"Language: {lang}  prefix: {LANG_PREFIX[lang]}  [{seen}]")

        lang_results = {}

        for split in ["dev", "test"]:
            path = data_dir / f"{lang}_{split}.tsv"
            if not path.exists():
                print(f"  [SKIP] {path} not found")
                continue

            examples = load_split(path)
            graphemes = [g for g, _ in examples]
            gold = [p for _, p in examples if p is not None]
            has_gold = len(gold) == len(graphemes)

            print(f"\n  Split : {split}  ({len(graphemes)} examples)")
            print(f"  Running greedy inference...")

            preds = predict_greedy(
                model,
                tokenizer,
                graphemes,
                lang,
                batch_size=args.batch_size,
                device=device,
            )

            # Write hypotheses
            hyp_path = output_dir / lang / f"{lang}_{split}.hyp"
            write_hypotheses(hyp_path, graphemes, preds)
            print(f"  Hypotheses → {hyp_path}")

            # Metrics (only if gold available)
            if has_gold:
                wer = compute_wer(gold, preds)
                per = compute_per(gold, preds)
                print(f"  WER : {wer:.2f}%")
                print(f"  PER : {per:.2f}%")
                lang_results[split] = {"wer": round(wer, 2), "per": round(per, 2)}

                print(f"\n  Samples:")
                print_samples(graphemes, gold, preds, n=args.n_samples)

        all_results[lang] = lang_results

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY — Zero-Shot Baseline")
    print(f"{'=' * 60}")
    print(f"  {'Lang':<8} {'Split':<8} {'WER':>8} {'PER':>8}")
    print(f"  {'-' * 36}")
    for lang, splits in all_results.items():
        for split, metrics in splits.items():
            print(
                f"  {lang:<8} {split:<8} {metrics['wer']:>7.2f}% {metrics['per']:>7.2f}%"
            )

    # Save results JSON
    results_path = output_dir / "baseline_zeroshot_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot ByT5 G2P baseline — SIGMORPHON 2021 Subtask 3"
    )
    parser.add_argument(
        "--data_dir",
        default="data/low",
        help="Directory with {lang}_train.tsv / {lang}_dev.tsv / {lang}_test.tsv files",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/baseline_zeroshot",
        help="Where to write hypotheses and results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Inference batch size",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="Number of sample predictions to print per split",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
