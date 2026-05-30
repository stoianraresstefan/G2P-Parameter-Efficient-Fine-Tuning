"""Display ordering, labels, and external reference values for tables/figures."""
from __future__ import annotations

# Config codes (method, with LoRA split by rank) in display order.
METHOD_ORDER = ["zeroshot", "frozen_head", "lora_r8", "lora_r16", "full_ft"]
LANG_ORDER = ["rum", "gre"]

DISPLAY_LANG = {"rum": "Romanian", "gre": "Greek"}

DISPLAY_METHOD = {
    "zeroshot": "Zero-shot",
    "frozen_head": "Frozen head",
    "lora_r8": "LoRA (r=8)",
    "lora_r16": "LoRA (r=16)",
    "full_ft": "Full FT",
}

# Reported SIGMORPHON 2021 low-resource baseline WERs (Ashby et al. 2021),
# shown as an external reference row in Table I. Not recomputed locally.
SIGMORPHON_BASELINE_WER = {"rum": 10.0, "gre": 21.0}

# Plot markers per config code.
MARKERS = {
    "zeroshot": "o",
    "frozen_head": "s",
    "lora_r8": "^",
    "lora_r16": "v",
    "full_ft": "D",
}
