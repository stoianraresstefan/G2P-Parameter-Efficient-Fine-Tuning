# IEEE Paper Outline

**Working title:** _LoRA for Multilingual Grapheme-to-Phoneme Conversion: Parameter-Efficient Adaptation of ByT5-G2P on Romanian and Modern Greek_

**Target length:** 7–8 pages (IEEE conference, 2-column), excluding references.
**Format:** IEEE Conference Template (https://www.ieee.org/conferences/publishing/templates.html)
**Deadline:** 2026-06-08

**Languages:** Romanian (`rum`) and Modern Greek (`gre`), both from the SIGMORPHON 2021 Shared Task 1 — Subtask 3 (low-resource, 1000 training words per language).

---

## Title, Authors, Affiliations

Standard IEEE block. Emails, university, course code.

---

## Abstract (≈200 words)

1. **Context** (1 sentence) — G2P is critical for TTS/ASR; pretrained multilingual models like ByT5-G2P exist but per-language adaptation relies on full fine-tuning.
2. **Gap** (1 sentence) — Parameter-efficient fine-tuning (PEFT) is standard in adjacent fields (LLMs, ASR) but unexplored for G2P.
3. **Contribution** (2 sentences) — We benchmark LoRA against full fine-tuning and a frozen-base reference on ByT5-G2P, using Romanian and Modern Greek from the SIGMORPHON 2021 low-resource subtask, measuring PER, WER, trainable parameter count, and forgetting on English CMUDict.
4. **Results** (2 sentences) — Headline number to fill after experiments, e.g., _"LoRA with rank 8 matches full FT within X% PER while training Y% of parameters and preserving English performance."_
5. **Takeaway** (1 sentence) — PEFT is a practical default for per-language G2P deployment.

> ✏️ Write the abstract LAST.

---

## I. Introduction (~1 page)

**Paragraph 1 — Motivation.** What G2P is, why it matters (TTS frontends, ASR lexicons). One concrete example.

**Paragraph 2 — The multilingual problem.** Pronunciation lexicons are expensive; pretrained multilingual G2P models help but still need per-language adaptation when data exists.

**Paragraph 3 — The cost problem.** Full fine-tuning per language scales poorly. PEFT is standard in LLMs and ASR; G2P has not adopted it.

**Paragraph 4 — Our contribution.**

- First study of LoRA for multilingual G2P.
- Benchmark on Romanian (Latin script, Romance) and Modern Greek (Greek script, Hellenic), both 1000-word low-resource splits from SIGMORPHON 2021 Subtask 3.
- Measure PER, WER, trainable parameter ratio, and catastrophic forgetting on English CMUDict.
- Code released.

**Paragraph 5 — Research question.**
_"Can LoRA match full fine-tuning of ByT5-G2P on low-resource languages while training under 1% of parameters, and how does it compare on forgetting?"_

**Paragraph 6 — Paper structure.** One sentence per remaining section.

---

## II. Related Work (~1.25 pages)

> 📌 Graded against 5 explicit rubric sub-questions. Use the subsections below — do **not** merge into flowing prose.

### A. Datasets and Benchmarks Used in G2P

CMUDict (~134K English ARPAbet pairs, the canonical English benchmark), NetTalk (older secondary), SIGMORPHON 2020/2021 (multilingual IPA, high/medium/low-resource subtasks), WikiPron (large Wikipedia-scraped). **Limitations:** no sentence context, no homograph-by-sense splits, IPA cleaning loses diacritics (Route 2019).

### B. Evaluation Methods and Their Limitations

PER and WER are standard. **Critique:** edit-distance metrics ignore phonetic distance (confusing /p/ with /b/ scores the same as /p/ with /z/). No standardized robustness benchmark — r-G2P (Zhao 2022) is a partial step.

### C. Architectures and Training Techniques

Brief timeline:

1. Joint-sequence n-gram models — Bisani & Ney 2008 (still a strong classical baseline).
2. LSTM seq2seq — Rao 2015, Yao & Zweig 2015 (removed alignment).
3. CNN seq2seq — Yolchuyeva 2019.
4. Transformer — Yolchuyeva 2020 (best size/accuracy tradeoff).
5. Pretrained byte-level multilingual — ByT5-G2P / CharsiuG2P, Zhu 2022 (our strong reference).
6. GBERT pretraining — Dong 2022.
7. LLM-based prompting — Fetrat Qharabagh 2024 (strong but expensive at inference).
8. PEFT in adjacent fields (Whisper, NMT): Hu 2021 (LoRA), Houlsby 2019 (adapters) — **not yet applied to G2P → the gap we address.**

### D. Field Shortcomings

- Low-resource adaptation remains weak; ByT5-G2P training is unstable on ~1800 training words (Zhu 2022).
- Per-language full fine-tuning wastes storage and compute.
- No standardized PEFT comparison exists for G2P.
- Robustness to spelling variation is poor (Zhao 2022).

### E. How the Works Relate to Each Other

A synthesis paragraph: classical n-gram → neural seq2seq → Transformer → pretraining/multilingual → contextual/LLM-based. Each generation removed a constraint (alignment, monolinguality, word-level scope, retraining cost). Our work continues this trajectory by removing the **per-language fine-tuning cost** constraint via PEFT.

---

## III. Methodology (~1.25 pages)

### A. Datasets

We use the **SIGMORPHON 2021 Shared Task 1 — Subtask 3 (low-resource)** data for two languages:

| Language     | Code  | Script | Family                   | Train | Dev | Test |
| ------------ | ----- | ------ | ------------------------ | ----- | --- | ---- |
| Romanian     | `rum` | Latin  | Indo-European / Romance  | 800   | 100 | 100  |
| Modern Greek | `gre` | Greek  | Indo-European / Hellenic | 800   | 100 | 100  |

_(Subtask 3 provides 1000 words per language, split 80/10/10 by the task organizers.)_

**Why this pair.** Both languages share the SIGMORPHON 2021 low-resource protocol — identical splits, evaluation script, and dataset size — making the cross-language comparison clean. They differ on three axes that exercise the model: **script** (Latin vs Greek), **language branch** (Romance vs Hellenic), and **orthographic patterns** (Romanian has palatalized sequences such as `lʷ`; Greek has consonant clusters and digraphs such as `μπ` → /b/). Both share an Indo-European root, so the inventory of target phonemes overlaps partially — a clean condition for studying PEFT generalization. Reported SIGMORPHON 2021 baseline WER: 10% on Romanian, 21% on Modern Greek, giving us measurable headroom on both.

**English CMUDict (v0.7b)** is used **only** for the catastrophic forgetting evaluation. We hold out a 1000-word random sample.

**Preprocessing.** Use SIGMORPHON 2021 splits as provided. Tokenize at byte level (ByT5 native). Filter pairs >50 characters. Normalize to NFC (matching SIGMORPHON convention).

### B. Baseline: Zero-Shot ByT5-G2P

- Load pretrained `charsiu/g2p_multilingual_byT5_small` from HuggingFace (~300M parameters).
- Evaluate **zero-shot** on Romanian and Greek test sets — no training, just inference with greedy decoding (CharsiuG2P authors note beam search does not help).
- Use the appropriate language-code prefix: `<rum>:` for Romanian, `<gre>:` for Greek.
- This satisfies the rubric's "baseline (e.g., ... pretrained model)" requirement.
- Establishes the "no adaptation" lower bound.

### C. Strong Reference: Full Fine-Tuning

- Same ByT5-G2P, full FT on each target language's 800-word training set.
- All 300M parameters updated. Establishes the "full adaptation" upper bound.

### D. Original Contribution: LoRA Fine-Tuning

We compare three adaptation strategies per language:

1. **Full fine-tuning** — all 300M parameters updated. Strong reference (Section III.C).
2. **LoRA** (Hu et al. 2021) — trainable rank-r matrices injected into attention Q, K, V, O projections. Test r ∈ {8, 16}, α = 2r.
3. **Frozen base + new LM head** — freeze all base weights, train only the output head.

All variants share LR, batch size, and epochs within each language for fair comparison. Implementation uses HuggingFace `peft`.

### E. Evaluation Protocol

- **Primary metrics:** Phoneme Error Rate (PER) and Word Error Rate (WER), computed with the official SIGMORPHON 2021 `evaluate.py` script for full comparability.
- **Efficiency metrics:** trainable parameter count (absolute + % of base), training wall-clock time.
- **Forgetting metric:** PER on the 1000-word English CMUDict held-out set, before vs. after adaptation. Measured for **best LoRA config** and **full FT only**, not every config.
- **Seeds:** 1 seed for main experiments; **3 seeds for the headline LoRA r=8 config on Romanian** to estimate variance.

### F. Implementation Details

HuggingFace `transformers` + `peft`. Kaggle 2× T4 GPUs, fp16 mixed precision. Code released on GitHub.

---

## IV. Experiments and Original Contribution (~1.25 pages)

Experimental matrix: { 4 configurations: full FT, LoRA r=8, LoRA r=16, frozen-head } × { 2 languages: Romanian, Greek } × { 1 seed } + 2 extra seeds on LoRA r=8 Romanian + 2 zero-shot baseline runs. **~12 total runs.**

### A. Baseline Results: Zero-Shot ByT5-G2P

Report PER/WER for Romanian and Greek with no adaptation. Sets the lower bound. Compare to SIGMORPHON 2021 baseline WERs (10% Romanian, 21% Greek).

### B. Strong Reference: Full Fine-Tuning

PER/WER per language after full FT of ByT5-G2P on the 800-word training split. Sets the upper bound.

### C. LoRA Results (Core Contribution)

Table of PER/WER per language × method × LoRA rank. Identify the best LoRA configuration per language.

### D. Pareto Analysis: Accuracy vs. Trainable Parameters

Scatter plot — x-axis: trainable parameters (log scale); y-axis: PER. Each method is one point per language. Highlight the Pareto frontier.

### E. Catastrophic Forgetting Analysis

For the best LoRA config and for full FT, evaluate the adapted model on the held-out English CMUDict sample. Bar chart with three bars per language: zero-shot baseline, LoRA-adapted, full-FT-adapted.

> Predicted finding: full FT noticeably degrades English; LoRA preserves it. Likely a clean win for PEFT.

### F. How Our Contribution Differs from Prior Work

- **vs. Zhu 2022 (ByT5-G2P):** they train from scratch; we adapt cheaply.
- **vs. Vesik 2020 (self-training):** orthogonal — PEFT is complementary.
- **vs. r-G2P (Zhao 2022):** orthogonal axis (robustness vs. parameter efficiency).
- **vs. LLM-G2P (Fetrat 2024):** they use frozen LLMs with prompting; we adapt small dedicated models with PEFT. Much cheaper at inference.

---

## V. Results and Discussion (~1.25 pages)

Heavy on visualizations:

**Table I — Main results.** PER and WER on Romanian and Greek test sets for: zero-shot baseline, frozen-head, LoRA (r=8), LoRA (r=16), full FT. Bold the best per row.

**Table II — Efficiency.** Trainable parameters (absolute and %), training time per method.

**Figure 1 — Pareto frontier.** PER vs. trainable parameters across configurations.

**Figure 2 — Forgetting.** English CMUDict PER for: original ByT5-G2P, after-LoRA, after-full-FT (Romanian and Greek separately).

### Discussion (prose, not bullets)

- Did LoRA match full FT on Romanian? On Greek? At what rank?
- Does the script difference (Latin Romanian vs Greek alphabet) matter for PEFT quality?
- How severe is catastrophic forgetting from full FT? Does that alone justify PEFT?
- Limitations: only two languages, only two LoRA ranks, mostly single seed, single base model (ByT5-small), only word-level G2P (no homograph disambiguation), no adapters comparison.
- Threats to validity: SIGMORPHON splits may have lemma-level train/test overlap — check.

---

## VI. Conclusion (~0.5 page)

Three paragraphs:

1. **Summary.** What we did, headline result (one number).
2. **Implications.** PEFT is a practical default for per-language G2P; storage cost of K languages drops from K × 300M to K × ~1M.
3. **Future work.** (a) Adapter modules and larger LoRA-rank sweep. (b) More languages spanning more scripts. (c) Combine PEFT with self-training (Vesik 2020). (d) Sentence-level G2P with homograph disambiguation (SoundChoice). (e) PEFT-based robustness (combine with r-G2P).

---

## References (separate page, not counted)

Target: **12–14 references**, exceeding the 10-paper rubric minimum.

1. Bisani & Ney (2008) — Joint-sequence models for G2P. _Speech Communication_.
2. Yao & Zweig (2015) — Seq2seq for G2P.
3. Rao et al. (2015) — LSTM G2P. ICASSP.
4. Yolchuyeva et al. (2019) — CNN G2P. Applied Sciences.
5. Yolchuyeva et al. (2020) — Transformer G2P. arXiv:2004.06338.
6. Zhu et al. (2022) — ByT5 multilingual G2P. Interspeech / arXiv:2204.03067.
7. Vesik et al. (2020) — One Model to Pronounce Them All. arXiv:2006.13343.
8. Route et al. (2019) — Multimodal multilingual G2P. ACL D19-6121.
9. Dong et al. (2022) — GBERT pretraining for G2P. arXiv:2201.10716.
10. Ashby et al. (2021) — Results of the Second SIGMORPHON Shared Task on Multilingual G2P. _Proc. SIGMORPHON 2021_.
11. Zhao et al. (2022) — r-G2P robustness. arXiv:2202.11194.
12. Fetrat Qharabagh et al. (2024) — LLM-Powered G2P. arXiv:2409.08554.
13. Wang et al. (2024) — Survey of G2P methods. MDPI Appl. Sci. 14:11790.
14. Hu et al. (2021) — LoRA. ICLR 2022 / arXiv:2106.09685.
15. Houlsby et al. (2019) — Adapters for NLP. ICML 2019. _(cited in Related Work and Future Work, not used experimentally)_

### Optional supplementary

- Vaswani et al. (2017) — Attention Is All You Need.

---

## Page Budget Recap

| Section                   | Approx. Pages |
| ------------------------- | ------------- |
| Title + Abstract          | 0.3           |
| I. Introduction           | 1.0           |
| II. Related Work          | 1.25          |
| III. Methodology          | 1.25          |
| IV. Experiments           | 1.25          |
| V. Results & Discussion   | 1.25          |
| VI. Conclusion            | 0.5           |
| Figures + Tables (inline) | ~1.0          |
| **Total**                 | **~7.5–8.0**  |

References go on a separate (un-counted) page.

---

## 9-Day Time Plan

| Day | Task                                                                                                                                                     |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Set up Kaggle env. Download SIGMORPHON 2021 Subtask 3 data for `rum` and `gre`. Load ByT5-G2P. Run zero-shot inference on both. Record baseline PER/WER. |
| 2   | Implement LoRA training loop with HuggingFace `peft`. Verify end-to-end on a small slice.                                                                |
| 3   | Run full FT + LoRA r=8 + LoRA r=16 + frozen-head on Romanian.                                                                                            |
| 4   | Same configs on Modern Greek.                                                                                                                            |
| 5   | Run 2 extra seeds on the headline config (LoRA r=8 Romanian). Run forgetting evaluation on CMUDict.                                                      |
| 6   | Generate all tables and figures. Save final results.                                                                                                     |
| 7   | Write Sections II, III, IV (related work, methodology, experiments).                                                                                     |
| 8   | Write Sections I, V, VI (intro, results & discussion, conclusion). Write abstract.                                                                       |
| 9   | Polish, proofread, build presentation slides. Buffer.                                                                                                    |

---

## Presentation Outline (6 min — separate deliverable, build Day 9)

~12 slides at ~30 sec/slide:

1. Title, team
2. The G2P problem in one example (use a Romanian word, e.g. "pornește" → "p o r n e ʃ t e")
3. Why multilingual G2P matters
4. The cost problem: full FT × K languages
5. Research question
6. Method overview (diagram: ByT5-G2P + LoRA injected)
7. Datasets (Romanian + Greek) + experimental setup
8. Main result table
9. Pareto frontier figure
10. Catastrophic forgetting figure
11. Limitations & future work
12. Conclusion + takeaway

Leave ~90s for Q&A buffer.

---

## Practical Writing Checklist

- [ ] Use IEEE Conference template (Overleaf has it pre-loaded).
- [ ] Answer the **5 rubric sub-questions** explicitly in Related Work subsections.
- [ ] Self-contained figure captions (a skim-reader should understand each figure).
- [ ] Bold the best result in each row of every table.
- [ ] Report variance for at least one headline config.
- [ ] Be explicit about scope limitations (better than reviewers noticing).
- [ ] Code repo link in footnote on page 1.
- [ ] Spellcheck IPA characters and Greek script — they break easily in LaTeX (use a Unicode-friendly font like `fontspec` with XeLaTeX, or escape via `\textipa{}`).
- [ ] Don't claim SOTA. Frame as a methodological study.
