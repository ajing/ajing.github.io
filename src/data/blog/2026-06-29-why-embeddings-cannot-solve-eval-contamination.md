---
author: Jing Lu
pubDatetime: 2026-06-29T09:00:00Z
title: "Why Embeddings Cannot Solve Eval-Set Contamination"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Pretraining
  - Evaluation
description: "A technical deep dive on why semantic embedding search is useful but insufficient for eval-set decontamination: leakage is about evaluation advantage, not just text similarity."
---

Semantic embeddings are useful for finding suspicious overlap between pretraining data and evaluation sets, but they cannot be the whole decontamination system. Eval-set contamination is not just a semantic-similarity problem. It is a question of whether a training document gives the model privileged access to the eval item, answer, solution path, benchmark format, or task-specific shortcut.

Embedding search answers a narrower question: "Is this text close to that text in representation space?" That is related to contamination, but it is not equivalent to contamination.

This essay explains why.

## 1. The target is not "similar text"

Suppose an eval item asks:

> What is the capital of Australia?

A training document says:

> Canberra became Australia's capital in the early twentieth century.

This is not a duplicate of the eval question. Depending on the embedding model, it may or may not be extremely close to the question. But it leaks the answer.

Now consider another training document:

> Australia has a federal parliamentary system, a large mining sector, and major cities including Sydney, Melbourne, Brisbane, Perth, Adelaide, and Canberra.

This may be semantically close to the question because it discusses Australia and Canberra. But it does not necessarily leak the benchmark item in the same way.

The difference is subtle but important. Contamination is not merely about topical closeness. It is about evaluation advantage.

That advantage can come from:

- the exact prompt;
- the correct answer;
- a solution derivation;
- a paraphrased version of the item;
- a translated version;
- a benchmark-specific template;
- a code-equivalent solution;
- a discussion of common benchmark mistakes;
- answer-key metadata;
- a model-generated synthetic copy.

Embeddings can help find some of these. They cannot reliably classify all of them.

## 2. Embedding similarity has no natural decontamination threshold

The most immediate engineering problem is threshold selection.

If the threshold is too strict, the system catches only near-duplicates. It misses paraphrases, translations, compressed explanations, code-equivalent solutions, and long documents with small leaking spans.

If the threshold is too loose, the system deletes too much legitimate data.

For example, an MMLU question about organic chemistry will be semantically close to many normal chemistry textbook passages. A HumanEval coding task will be semantically close to many normal StackOverflow answers and GitHub examples. A GSM8K arithmetic word problem will be semantically close to many elementary math worksheets.

Deleting all nearby material may make the model worse at the underlying domain. But keeping all nearby material may preserve benchmark leakage.

This is the central threshold dilemma:

> A contamination detector needs to distinguish "this teaches the domain" from "this gives away the exam."

Generic embeddings were not designed to make that distinction.

## 3. The unit of matching is unstable

Embedding search depends heavily on chunking.

At document level, leakage can disappear. A 20,000-token webpage may contain one paragraph that quotes an eval item. The full-document embedding mostly represents the broader page topic, not the leaking span.

At paragraph level, recall improves, but cost and false positives increase.

At sentence level, the system may catch answer sentences, but it can lose context. A sentence like "The answer is B" is useless without the surrounding question. A sentence like "Therefore, the function returns the length of the longest substring" may only be meaningful if linked to a coding prompt.

At sliding-window level, recall improves again, but the number of comparisons explodes.

This means "use embeddings" is not a complete algorithm. The real algorithm must specify:

- chunk size;
- overlap between chunks;
- benchmark serialization;
- whether answer choices are embedded with the prompt;
- whether labels and explanations are embedded separately;
- how to handle code blocks;
- how to handle tables, PDFs, markdown, and comments;
- how to aggregate scores across chunks;
- what action to take after a hit.

Small choices here can change the contamination report.

## 4. Embeddings can miss answer-only leakage

Many benchmark items can be leaked without repeating the prompt.

For factual QA, a document containing the answer fact can help. For multiple-choice exams, a study guide may list the right concept without reproducing the exact question. For code, a repository may contain a functionally equivalent implementation without the benchmark prompt. For math, a solution page may show the derivation with different variable names.

Embedding the eval prompt and searching the corpus may miss these cases because the leaking document is not necessarily similar to the prompt. It may be similar to the answer, the explanation, or the latent solution concept.

A stronger system embeds multiple views of each benchmark item:

- prompt only;
- prompt plus choices;
- answer only;
- explanation only;
- canonical solution;
- normalized code;
- generated paraphrases;
- generated translations;
- task template or reasoning skeleton.

But once we do this, we are no longer "just using embeddings." We are building a benchmark-aware retrieval and review system.

## 5. Embeddings struggle with code contamination

Code contamination is especially hard.

Consider a benchmark prompt:

> Write a function that returns the longest common prefix among a list of strings.

A GitHub file might contain:

```python
def prefix(xs):
    if not xs:
        return ""
    result = xs[0]
    for item in xs[1:]:
        while not item.startswith(result):
            result = result[:-1]
    return result
```

This file may not contain the benchmark wording. It may not mention "HumanEval." It may not share many tokens with the prompt. But it solves the same task.

A generic embedding model may find it, or it may not. Even if it finds it, the score may be similar to many legitimate code examples. The hard question is not semantic closeness. The hard question is behavioral equivalence.

Code decontamination may need:

- repository-level source quarantine;
- exact and fuzzy prompt matching;
- function-name and docstring matching;
- AST-level similarity;
- normalized token matching;
- unit-test behavior comparison;
- import/dependency matching;
- known benchmark mirror detection;
- solution-template matching.

Embeddings are useful, but they are only one signal in a domain-specific system.

## 6. Embeddings struggle with math contamination

Math contamination has a different failure mode.

Two problems can be surface-different but structurally identical:

> Alice buys 3 notebooks at 4 dollars each and 2 pens at 1 dollar each. How much does she spend?

and:

> A store sells 3 packs of paper for 4 dollars each and 2 erasers for 1 dollar each. What is the total cost?

The names and objects changed, but the computation is the same:

> 3 * 4 + 2 * 1

Should this count as contamination? It depends on the evaluation claim.

If the benchmark claims to test arithmetic skill, then seeing many similar examples may be normal training. If the exact numeric structure and reasoning template were generated from the benchmark, then it may be leakage. If the model saw the same problem with renamed entities, then the benchmark is less independent.

Embeddings cannot resolve this policy question. A math contamination detector may need equation extraction, template matching, answer-path comparison, and generated counterfactual rewrites.

## 7. Embedding models can themselves be contaminated

There is also a measurement issue. The embedding model used for cleaning may have been trained on internet-scale data. It may already have seen the same public benchmarks.

That does not make embeddings useless. But it means the detector is not a perfectly independent instrument. A contaminated embedding model can encode benchmark-specific associations, benchmark wording, or answer relationships.

In practice, this suggests two safeguards:

- use multiple retrieval signals, not only one embedding model;
- treat embedding hits as candidates for policy review rather than final truth.

## 8. Embedding search creates false confidence

The most dangerous failure mode is not that embeddings are bad. It is that they look sophisticated enough to create confidence.

A team might say:

> We embedded all eval items, searched the pretraining corpus, removed matches above 0.85 cosine similarity, and therefore the data is clean.

This statement hides many unresolved questions:

- Why 0.85?
- Was matching done at document, paragraph, or sentence level?
- Were answers embedded separately?
- Were solutions embedded?
- Were translations checked?
- Were benchmark mirrors and GitHub repos quarantined by source?
- Were code tasks checked behaviorally?
- Were retained semantic-near matches sampled manually?
- Did the team evaluate on rewritten or newly created holdouts?
- What residual risk remains?

Without those details, embedding decontamination is not an audit. It is a heuristic.

## 9. What embeddings are good for

The right conclusion is not "do not use embeddings." The right conclusion is "do not use only embeddings."

Embeddings are useful for:

- finding paraphrases missed by n-gram filters;
- discovering benchmark discussions that do not quote prompts exactly;
- clustering suspicious documents for review;
- finding translated or summarized variants when paired with multilingual models;
- prioritizing high-risk examples;
- producing candidate matches for LLM-assisted or human review.

They are especially helpful as a recall layer after exact and near-duplicate matching.

But embeddings should feed into a broader system with reason codes, thresholds, sampling, and domain-specific checks.

## 10. A better eval decontamination design

A more serious pipeline might look like this.

### Step 1: Build a benchmark registry

For every eval item, store:

- item ID;
- prompt;
- answer choices;
- correct label;
- canonical answer;
- explanation;
- solution code if applicable;
- release date;
- source URL;
- task family;
- benchmark split;
- license;
- known mirrors.

### Step 2: Generate multiple searchable views

For each item, construct:

- normalized prompt;
- prompt plus answer choices;
- answer-only view;
- explanation view;
- code-only view;
- extracted equations;
- generated paraphrases;
- generated translations for high-risk benchmarks;
- task-template summaries.

### Step 3: Run layered retrieval

Use:

- exact hashes;
- normalized substring matching;
- n-gram overlap;
- MinHash / SimHash;
- source and URL quarantine;
- embedding retrieval;
- code AST similarity;
- math template matching;
- image perceptual hashes where relevant.

### Step 4: Classify match severity

Do not reduce everything to one similarity score. Use severity labels:

- exact prompt match;
- prompt plus answer match;
- answer-only match;
- explanation match;
- paraphrase candidate;
- translation candidate;
- code-equivalent candidate;
- source-level benchmark mirror;
- weak topical similarity.

These labels matter because removal policy should differ by severity.

### Step 5: Choose policy actions

Possible actions:

- remove document;
- remove chunk;
- quarantine source;
- keep but flag;
- sample for manual review;
- report as residual risk.

The action should depend on the benchmark's importance, the severity of the match, the source reliability, and the cost of false positives.

### Step 6: Validate with model behavior

Data-side checks should be paired with behavior-side checks:

- compare performance on original vs rewritten items;
- evaluate on fresh time-split benchmarks;
- test whether the model can reproduce benchmark wording;
- test whether performance is unusually high on suspected contaminated subsets;
- inspect confidence and generation traces where possible.

This still does not prove zero contamination, but it gives a more honest picture of risk.

## 11. The key distinction

The key distinction is this:

> Semantic similarity asks whether two texts are close.
>
> Eval contamination asks whether training exposure compromises the independence of measurement.

Those are not the same question.

Sometimes a semantically close document is harmless. Sometimes a semantically distant document leaks the answer. Sometimes the problem is not a document at all, but a source repository, a benchmark mirror, a synthetic data chain, or a repeated task template.

That is why eval-set decontamination cannot be solved by embeddings alone.

## 12. Conclusion

Embeddings should be part of modern pretraining decontamination. They are too useful to ignore, especially for paraphrases and weakly transformed benchmark copies.

But embedding-only cleaning is insufficient because contamination is target-conditioned, domain-specific, and policy-dependent. It is about unfair evaluation advantage, not merely semantic closeness.

The right standard is an auditable, layered decontamination system: exact matching, near-duplicate detection, source quarantine, embedding retrieval, task-specific similarity, severity classification, and residual-risk reporting.

In other words, embeddings are a search tool. Decontamination is an evaluation-governance problem.

## References

- [Rethinking Benchmark and Contamination for Language Models with Rephrased Samples](https://arxiv.org/abs/2311.04850)
- [Investigating Data Contamination in Modern Benchmarks for Large Language Models](https://arxiv.org/html/2311.09783v2)
- [A Survey on Data Contamination for Large Language Models](https://arxiv.org/html/2502.14425v2)
- [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/html/2406.11794v1)
