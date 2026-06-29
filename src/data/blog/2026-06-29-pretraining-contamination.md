---
author: Jing Lu
pubDatetime: 2026-06-29T08:00:00Z
title: "Pretraining Contamination: Why Don't Train on the Test Set Became Hard"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Pretraining
  - Evaluation
description: "A practical introduction to LLM pretraining contamination: why benchmark leakage is not ordinary deduplication, how public evals leak into web-scale corpora, and how layered decontamination pipelines reduce risk."
---

Pretraining contamination is not a small data-cleaning mistake. It is a measurement problem. Once language models are trained on internet-scale corpora, public benchmarks, answer keys, code solutions, exam explanations, leaderboard discussions, and synthetic copies of all of the above can enter the training mixture. The result is that benchmark scores may measure a mixture of generalization, memorization, benchmark familiarity, and data-pipeline luck.

The old rule was simple: do not train on the test set. In modern LLM pretraining, that rule becomes much harder to enforce because there is no single neat training set and no single neat test set. There is a large, messy web-scale corpus on one side and a collection of public evaluation artifacts on the other. The hard question is no longer "did test.csv accidentally get included?" The hard question is "did the model see enough information, in any form, to gain an unfair advantage on this evaluation?"

## 1. What contamination means

In classical supervised learning, contamination usually means train-test overlap. A row from the test set appears in the training set. That is bad, but it is at least easy to define.

For LLMs, contamination is broader. A benchmark item can leak through:

- the exact question;
- the exact question plus answer;
- a benchmark solution explanation;
- a translated version;
- a paraphrased version;
- a code solution in a GitHub repository;
- a tutorial that quotes the benchmark;
- a forum post discussing the answer;
- synthetic instruction data generated from a contaminated model;
- a benchmark-specific trick, template, or answer distribution.

This is why contamination is not the same thing as deduplication. Deduplication asks whether two training examples are redundant with each other. Decontamination asks whether a training example gives the model privileged access to an evaluation item.

A document can be unique and still contaminated. A single blog post explaining five benchmark questions is not a duplicate of anything else. But if it appears in pretraining, it can still undermine the benchmark.

## 2. Why pretraining makes the problem worse

Pretraining makes contamination harder for three reasons.

First, the data is huge. A modern pretraining corpus may include web pages, Common Crawl snapshots, books, papers, GitHub repositories, Q&A sites, documentation, scraped PDFs, subtitles, code comments, and synthetic data. At that scale, public benchmarks are likely to appear somewhere.

Second, the benchmark ecosystem is public and self-replicating. The more important a benchmark becomes, the more people write about it. They publish solutions, reproduce questions, make study guides, discuss edge cases, create translated copies, upload notebooks, and build "benchmark practice" datasets. Famous benchmarks leak because fame creates copies.

Third, synthetic data creates indirect contamination. Suppose Model A saw a benchmark. Model A then generates instruction data. Model B trains on that synthetic data. Model B may inherit benchmark-like examples even if the original benchmark file never appears in Model B's raw corpus.

This means contamination is not only a historical artifact of sloppy dataset construction. It is a continuing ecosystem problem.

## 3. Different types of contamination

The most useful way to reason about contamination is to separate several levels.

### Exact input contamination

The eval prompt appears verbatim in the training corpus. This is the easiest case to detect.

Example:

> Eval: "Which planet is known as the Red Planet?"
>
> Training data: "Which planet is known as the Red Planet?"

This can often be caught with normalized exact matching, n-gram matching, MinHash, or other near-duplicate methods.

### Input-and-label contamination

The model sees both the question and the answer.

Example:

> "Which planet is known as the Red Planet? Answer: Mars."

This is more severe because the training document contains the mapping the benchmark is trying to test.

### Partial contamination

Only part of the benchmark item appears. For example, a reading-comprehension passage appears without the exact question, or a math problem appears without the final answer.

Partial contamination is harder to interpret. It may help a lot, a little, or not at all. But it still matters because the evaluation is no longer fully independent.

### Semantic contamination

The model sees a paraphrase, translation, equivalent code task, or near-equivalent reasoning problem.

Example:

> Eval: "A train travels 60 miles in 1.5 hours. What is its average speed?"
>
> Training data: "If a car covers 120 kilometers in 3 hours, calculate its average speed."

This may not be the same literal question. But if the benchmark is supposed to test whether the model can infer rate = distance / time, then enough near-equivalent examples can change what the score means.

### Benchmark-format contamination

The model may learn the quirks of a benchmark without memorizing individual items. It may learn common phrasing, answer styles, multiple-choice patterns, common distractors, or prompt templates.

This matters because many benchmarks are not pure capability tests. They are artifacts with recognizable styles.

### Post-training contamination

Even if pretraining is clean, contamination can enter later through supervised fine-tuning, RLHF data, benchmark-focused instruction datasets, or model-generated explanations. A clean base model can become contaminated during alignment or instruction tuning.

## 4. Why this affects benchmark trust

Contamination does not automatically mean a benchmark score is fake. The effect depends on the benchmark, the model, the number of exposures, the training stage, and the type of overlap.

For factual QA, seeing the answer can directly improve performance.

For code generation, seeing a canonical solution or a near-identical GitHub task can help the model reproduce the right structure.

For math, seeing many templated variants can help the model learn a shallow pattern rather than solve a fresh problem.

For broad knowledge benchmarks, exposure to related educational material may be legitimate pretraining, while exposure to the exact benchmark item is not.

This is the core ambiguity: pretraining is supposed to teach the model from the world, and benchmarks are also drawn from the world. The goal is not to remove all knowledge related to an evaluation topic. The goal is to avoid giving the model access to the exam itself.

That boundary is not always clean.

## 5. Why simple cleaning is not enough

Many teams start with exact matching or n-gram overlap. These methods are valuable. They are scalable, explainable, and good at catching obvious leaks.

But they are incomplete.

They can miss:

- paraphrases;
- translations;
- reordered multiple-choice options;
- reformatted code;
- solution explanations without the original prompt;
- benchmark discussions with abbreviated prompts;
- long documents where only one paragraph leaks;
- synthetic variants generated from contaminated sources.

On the other side, aggressive semantic filtering can remove too much. If every document semantically close to an MMLU question is deleted, the training set may lose normal educational material. If every document close to a coding benchmark is deleted, the model may lose legitimate programming examples.

This creates a real engineering tradeoff. Too little filtering leaves leakage. Too much filtering damages the training distribution and may remove the very knowledge the benchmark is meant to test.

## 6. What a serious decontamination pipeline actually does

A mature decontamination system is not one classifier and not one embedding index. It is a layered pipeline with explicit policies. The important engineering idea is that each layer catches a different failure mode.

### Step 1: Build a benchmark registry

The first artifact is a benchmark registry. For every benchmark you intend to report, store the item ID, split, release date, prompt, answer choices, correct label, explanation, source URL, license, and task family.

For multiple-choice and factual QA, store several searchable views:

- prompt only;
- prompt plus all choices;
- prompt plus correct answer;
- answer only;
- explanation only;
- metadata such as category, source, and original URL.

This detail matters because answer-only or prompt-plus-answer retrieval can find leaks that prompt-only search misses. Deng et al. report that concatenating question and label improved retrieval efficiency for contamination detection on benchmarks such as MMLU and TruthfulQA ([Deng et al., 2023](https://arxiv.org/html/2311.09783v2)).

### Step 2: Canonicalize before matching

Both benchmark items and candidate training documents should be normalized before matching. At minimum:

- lowercase where appropriate;
- normalize whitespace, punctuation, unicode variants, and HTML entities;
- remove boilerplate navigation text;
- canonicalize Markdown and PDF extraction artifacts;
- serialize multiple-choice examples in a stable format;
- format code consistently;
- keep source URL and crawl timestamp attached to every chunk.

This is boring infrastructure, but it is the difference between a real detector and a demo. Without canonicalization, tiny formatting differences create false negatives.

### Step 3: Use exact and n-gram matching as the first wall

Exact and n-gram matching should still be the first wall because it is cheap, scalable, and interpretable. This is the family of methods used in early large-model decontamination work: later contamination literature summarizes GPT-3 as using a 13-gram-style strategy, while PaLM split examples into clean and contaminated subsets when at least 70% of the 8-grams in the question, prompt, or target appeared in training data ([Deng et al., 2023](https://arxiv.org/html/2311.09783v2); [PaLM](https://jmlr.org/papers/v24/22-1144.html)).

The practical version is:

- hash exact benchmark strings and normalized strings;
- run substring search for long benchmark spans;
- run n-gram overlap at the chunk level;
- use MinHash or SimHash for near-duplicates;
- record the exact matched span, benchmark ID, document ID, source URL, score, and action.

The action should be aggressive for exact prompt-plus-answer overlap. Remove the chunk or quarantine the whole document, depending on how the corpus is assembled.

### Step 4: Quarantine known bad sources

Some leakage is source-level, not item-level. If a GitHub repository is a benchmark mirror, a Kaggle notebook contains benchmark solutions, or a website exists to publish answer keys, matching item-by-item is too fragile. The safer policy is to quarantine the source.

Useful source-level signals include:

- URLs containing benchmark names;
- GitHub repositories with benchmark files, answer keys, or leaderboard scripts;
- notebooks titled as benchmark solutions;
- benchmark README mirrors;
- scraped forums or study guides discussing exact eval items;
- synthetic datasets explicitly generated from benchmark prompts.

This is especially important for code benchmarks. A solution repository may not repeat the exact prompt, but it can still contain behaviorally equivalent solutions.

### Step 5: Add semantic retrieval, but use it as triage

Embedding search is valuable, but it should be a candidate generator, not the final judge. It can surface paraphrases, translations, summaries, and explanation pages that n-gram matching misses. But it also brings false positives because many legitimate educational documents are semantically close to benchmark questions.

Yang et al. show the key failure mode: paraphrased or translated benchmark samples can bypass string-matching decontamination, and if those variants remain in training, a 13B model can overfit the benchmark and reach drastically inflated performance ([Yang et al., 2023](https://arxiv.org/abs/2311.04850)).

So the practical policy should be:

- use embeddings to retrieve top-k candidates for each benchmark item;
- classify the candidate as exact, answer-only, explanation, paraphrase, translation, code-equivalent, weak topical match, or harmless;
- sample borderline cases for human or LLM-assisted review;
- remove high-confidence semantic leaks;
- report weak semantic-near matches as residual risk rather than pretending they do not exist.

### Step 6: Use task-specific detectors

Different benchmark families need different detectors.

For code:

- compare normalized code tokens;
- compare AST structure;
- quarantine known benchmark mirrors;
- check function names, docstrings, and unit-test behavior;
- look for canonical solutions and near-equivalent implementations.

For math:

- extract equations;
- normalize variable names and entities;
- compare solution templates;
- detect same-number or same-operation variants;
- distinguish "teaches arithmetic" from "renamed benchmark item."

For reading comprehension:

- match passages separately from questions;
- search source documents;
- check whether the answer sentence appears in training;
- detect whether the same passage appears with different questions.

For multimodal benchmarks:

- use perceptual hashes for images;
- check captions, alt text, filenames, and surrounding HTML;
- treat image-text pairs as the unit of contamination.

### Step 7: Add evaluation-side defenses

Data cleaning alone is not enough, especially once public benchmarks become famous. Evaluation itself should become more robust.

One direction is time-sensitive evaluation. LatestEval, for example, creates reading-comprehension evaluations from recent texts so the benchmark is less likely to overlap with older pretraining corpora; its pipeline gathers recent texts, identifies key information, and constructs questions while removing existing answers from the context ([LatestEval](https://arxiv.org/abs/2312.12343)).

Other evaluation-side defenses include:

- private holdout sets;
- one-time exams;
- benchmark item rotation;
- adversarial rewrites of public items;
- measuring score drop from original to rewritten versions;
- reporting results separately on suspected contaminated and clean subsets.

This matters because decontamination can reduce risk, but it cannot prove the model has never seen related material.

### Step 8: Publish an auditable report

A serious release should include a decontamination report. DCLM is a useful model here: it releases decontamination tooling and asks submissions to disclose a decontamination report rather than treating contamination as a private implementation detail ([DCLM](https://arxiv.org/html/2406.11794v1)).

The report should include:

- protected benchmark list and versions;
- benchmark registry schema;
- training-data sources and crawl windows;
- canonicalization rules;
- exact, n-gram, near-duplicate, semantic, and task-specific matching methods;
- thresholds and why they were chosen;
- counts of removed documents, chunks, and tokens;
- examples of removed and retained borderline matches;
- residual-risk categories;
- performance on clean vs. suspicious subsets where available;
- whether fresh, private, or time-sensitive evaluations were used.

The goal is not to claim "zero contamination." The goal is to make the remaining uncertainty visible.

## 7. The deeper issue: evaluation governance

The phrase "data decontamination" makes this sound like cleaning. But the deeper issue is governance.

If benchmark scores are used to compare labs, rank models, advertise capabilities, allocate funding, or claim progress toward reasoning, then the benchmark must measure what people think it measures.

Contamination threatens that measurement.

This is why decontamination should become an auditable artifact of pretraining. A model release should not only say "we cleaned the data." It should explain:

- which benchmarks were protected;
- which matching methods were used;
- which data sources were quarantined;
- what thresholds were chosen;
- how many documents or tokens were removed;
- what kinds of residual contamination may remain;
- whether rewritten or fresh evaluations were used as a check.

The goal is not to prove there is zero leakage. At web scale, that may be impossible. The goal is to make contamination risk measurable, comparable, and honest.

## 8. Conclusion

Pretraining contamination is hard because language models learn from the same public internet that benchmarks live on. The more public and important a benchmark becomes, the more likely it is to appear in training data directly or indirectly.

This does not mean benchmarks are useless. It means benchmark scores need provenance, context, and humility.

The old rule still matters: do not train on the test set. But for LLMs, that rule has become an engineering and scientific discipline. It requires benchmark registries, source tracking, multi-stage matching, semantic review, domain-specific checks, and transparent reporting.

Decontamination is not just about cleaner data. It is about protecting the meaning of evaluation.

## References

- [A Survey on Data Contamination for Large Language Models](https://arxiv.org/html/2502.14425v2)
- [Investigating Data Contamination in Modern Benchmarks for Large Language Models](https://arxiv.org/html/2311.09783v2)
- [Rethinking Benchmark and Contamination for Language Models with Rephrased Samples](https://arxiv.org/abs/2311.04850)
- [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/html/2406.11794v1)
- [PaLM: Scaling Language Modeling with Pathways](https://jmlr.org/papers/v24/22-1144.html)
- [LatestEval: Addressing Data Contamination in Language Model Evaluation through Dynamic and Time-Sensitive Test Construction](https://arxiv.org/abs/2312.12343)
