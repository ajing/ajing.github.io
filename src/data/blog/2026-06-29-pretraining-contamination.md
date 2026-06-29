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
description: "A practical introduction to LLM pretraining contamination: why benchmark leakage is not ordinary deduplication, how public evals leak into web-scale corpora, and what an auditable decontamination pipeline should report."
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

## 6. What a serious decontamination pipeline looks like

A mature decontamination system should be layered.

First, maintain a benchmark registry. The team needs canonical copies of eval prompts, labels, answer choices, solution text, metadata, release dates, and licensing information.

Second, normalize both training data and benchmark data. Lowercasing, whitespace cleanup, punctuation normalization, code formatting, and canonical multiple-choice serialization all matter.

Third, run exact and near-duplicate matching. This includes hashes, n-gram overlap, MinHash, SimHash, and URL/source-based quarantine.

Fourth, add semantic retrieval. Embeddings can help find paraphrases and related passages, but they should be treated as a recall-oriented signal, not a final judge.

Fifth, add domain-specific checks. Code benchmarks may need AST similarity, import/function-name matching, repository-level quarantine, or unit-test behavior comparison. Math benchmarks may need equation normalization and template matching. Vision benchmarks may need perceptual image hashes.

Sixth, apply a removal policy. Exact question-and-answer overlap should usually be removed. Weak semantic overlap may be flagged, sampled, or quarantined depending on risk.

Seventh, publish a residual-risk report. The right answer is rarely "perfectly clean." A more honest report says what was checked, what was removed, what was retained, and where uncertainty remains.

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
