---
author: Jing Lu
pubDatetime: 2026-09-02T09:00:00-07:00
title: "When Does Model Souping Work for LLMs?"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Post Training
  - Evaluation
description: "A practical guide to when LLM weight averaging and model merging work, why they fail, how methods such as Task Arithmetic, TIES, DARE, and LoRA merging differ, and how to evaluate a merge before deployment."
---

Model souping has an unusually attractive promise: take several fine-tuned
models, average or combine their weights, and get one stronger model without
another full training run or the inference cost of an ensemble.

Sometimes that promise is real. Sometimes the merged model quietly regresses.
And sometimes an apparently sophisticated recipe mostly moves a strong base
model away from a region that was already working.

The useful mental model is not:

> Model merging automatically forms the union of everything the source models
> know.

It is:

> Model souping is weight-space interpolation. It works best when the source
> models still live in a compatible coordinate system and a shared low-loss
> region.

That distinction explains most of the empirical results. Same-task models
fine-tuned from the same checkpoint are often mergeable. Models trained on
clearly separated tasks from the same base can sometimes be composed. A random
collection of community checkpoints—even from the same model family—is much
less predictable. Models from different architectures or tokenizers are not
ordinary soup ingredients at all.

This post develops that framework, surveys the main techniques, reconciles the
positive and negative evidence, and ends with a concrete merge-evaluation
protocol.

## The short answer

Model souping is most likely to work when all of the following are true:

1. The models share the exact same pretrained checkpoint, architecture,
   tokenizer, vocabulary, special tokens, and parameterization.
2. Fine-tuning has not moved them too far from the base or into different loss
   basins.
3. Every ingredient is independently competent; the soup is not being asked to
   rescue failed runs.
4. The source models have useful diversity, but their objectives are not
   strongly contradictory.
5. The final update remains close enough to the base model to preserve general
   capabilities.
6. Membership, coefficients, sparsity, and global scale are selected on a
   validation set with separate capability and safety guardrails.

A practical risk map looks like this:

| Setting                                                  | Expected reliability     | Start with                                                               | Main risk                                              |
| -------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------ |
| Nearby checkpoints from one training trajectory          | High                     | Checkpoint averaging, EMA, SWA, LAWA                                     | Averaging across incompatible training phases          |
| Same task, same base, different seeds or hyperparameters | Medium-high              | Greedy soup; uniform only after filtering                                | One bad run can poison the average                     |
| Distinct tasks, same base                                | Conditional              | Normalized Task Arithmetic, then TIES or DARE variants                   | Task-vector conflict and forgetting                    |
| Public checkpoints with uncertain training histories     | Medium-low               | Conservative Task Arithmetic plus base interpolation                     | Gains are usually small; advanced methods can be worse |
| LoRAs from the same base                                 | Conditional              | Merge materialized deltas; coefficient search; KnOTS when bases misalign | Low-rank subspaces are not automatically aligned       |
| Same architecture but independently pretrained bases     | Low                      | Align permutations or representations first                              | Neurons and features do not share coordinates          |
| Different architectures, hidden sizes, or tokenizers     | Not a dense-soup problem | Routing, MoE, output ensembles, or distillation                          | There is no well-defined elementwise correspondence    |

The closer a project is to the top of the table, the more model souping behaves
like variance reduction. The closer it is to the bottom, the more it becomes an
alignment, routing, or retraining problem.

## What exactly is being merged?

The narrow form of model soup is a weighted average of checkpoints:

$$
\theta_{\text{soup}} = \sum_i \alpha_i \theta_i,
\qquad \sum_i \alpha_i = 1.
$$

A **uniform soup** gives every model equal weight. A **greedy soup** starts from
the best validation model and adds another model only when the combined
checkpoint improves the validation objective. A learned soup searches or
optimizes the coefficients.

Cross-task merging is usually written relative to a common base model
$\theta_0$. Each fine-tune defines a task vector:

$$
\Delta_i = \theta_i - \theta_0.
$$

The merged model is then:

$$
\theta_{\text{merge}}
= \theta_0 + \lambda \sum_i \alpha_i \Delta_i.
$$

The coefficients control the contribution of each expert. The global scale
$\lambda$ controls how far the result moves away from the base.

This is not the same as an output ensemble. An ensemble runs multiple models at
inference time and combines their predictions. It is also not distillation,
which trains a student on teacher outputs. Souping tries to compress the
combination into one parameter set with the same inference footprint as a
single dense model.

## Why averaging can work

### Shared pretraining creates a common coordinate system

Neural networks have permutation symmetries. Two networks can implement
similar functions while assigning those functions to different neurons. If
they were trained independently, elementwise averaging may combine unrelated
features.

Shared pretraining changes the geometry. Fine-tunes that branch from the same
long pretrained trajectory tend to preserve neuron correspondence, feature
organization, and proximity in parameter space. Transfer-learning and linear
mode-connectivity studies show that pretrained-then-fine-tuned models are much
more likely to be connected by low-loss paths than independently trained
models.

This is why “same architecture” is not enough. The exact base checkpoint often
matters more than the model-family name.

For LLMs, coordinate compatibility also includes details that are easy to
overlook:

- tokenizer and vocabulary order;
- embedding and output-head tying;
- special-token IDs;
- chat templates;
- positional encoding and RoPE scaling;
- normalization conventions;
- added heads;
- LoRA target modules, ranks, and scaling.

A merging tool may be able to reconcile tensor shapes or construct a union
vocabulary. That means the recipe can run. It does not prove that the resulting
weights have compatible semantics.

### The interpolation path needs to stay in a low-loss region

For two models, define a linear interpolation:

$$
\theta(\alpha) = (1-\alpha)\theta_1 + \alpha\theta_2,
\qquad \alpha \in [0,1].
$$

If validation loss or perplexity rises sharply between the endpoints, the two
models are separated by a loss barrier. Their midpoint is a poor soup
candidate.

The original [Model Soups paper](https://proceedings.mlr.press/v162/wortsman22a.html)
connects successful averaging to models occupying a common low-error basin.
When outputs vary approximately linearly along the path and the loss curvature
is favorable, the weight average can recover part of the benefit of a logit
ensemble without running every model.

But flatness is not sufficient. A convex interpolation curve can still be best
at an endpoint. Pairwise low-loss connectivity does not prove that the entire
multi-model convex hull is safe. And autoregressive generation, long reasoning
chains, and safety behavior are more complicated than a classification loss.

### Good soups need controlled diversity

If the models are nearly identical, their average is nearly identical and the
gain is negligible. Different seeds, data orders, moderate learning-rate
changes, and dropout can create complementary errors.

If the models are too different, however, averaging becomes destructive.
Aggressive learning rates, long fine-tuning runs, continued pretraining on
different domains, or large reinforcement-learning updates can push
checkpoints into different regions.

The LLM reward-model study
[WARM](https://proceedings.mlr.press/v235/rame24a.html) makes this trade-off
explicit. Its reward models share the same pretrained/SFT model and a
compatible reward head, while diversity comes from data order, learning rate,
dropout, or nearby checkpoints. Weight averaging improves robustness to
distribution shift and label noise in that controlled setting.

The lesson is not “maximize diversity.” It is “create diversity without
breaking connectivity.”

## The evidence is more conditional than the leaderboard narrative

### The original NLP result contains an important warning

The main Model Soups results came from vision models, with preliminary BERT and
T5 experiments. The NLP appendix is especially useful because it shows how
uniform averaging fails when the candidate pool contains poor runs.

Across five BERT/T5 sizes and four GLUE tasks—20 model-dataset combinations—the
uniform soup was worse than the best single model in all 20. The hyperparameter
range was intentionally broad, including learning rates from $10^{-6}$ to
$10^{-3}$, multiple batch sizes, and multiple epoch counts without early
stopping. Some ingredients were terrible. Greedy soup filtered them and
strictly improved over the best single model in 10 of the 20 combinations,
tying in the rest.

The right conclusion is not that uniform averaging never works for language
models. It is that “same base” is not enough. Uniform soup assumes the
ingredients are individually good.

### Controlled experts and community checkpoints are different data regimes

Methods such as TIES and DARE often look strong in controlled evaluations where
experts have distinct roles and share a known base. A more recent systematic
study tested a harder setting: 12 public fine-tuned checkpoints for each of
Llama 3.2 3B, Llama 3.1 8B, Qwen3 4B, and Qwen3 8B, evaluated across 16
benchmarks.

In that
[in-the-wild study](https://arxiv.org/abs/2511.21437), Task Arithmetic was the
only evaluated method that reliably produced constructive interference. The
average improvements were generally below one point. With at least four source
models, Task Arithmetic beat the best individual checkpoint for three of the
four base models; the unusually strong Llama 8B expert was never surpassed.
TIES, Model Stock, and subspace methods did not show a consistent advantage,
and some degraded as more models were added.

The mechanism is revealing. Normalized Task Arithmetic averaged many nearly
orthogonal updates toward a small net vector, keeping the result close to the
base. More aggressive subspace transformations produced much larger parameter
displacements—often associated with worse performance. In heterogeneous
settings, simple cancellation can be a feature rather than a bug.

The same paper found much less interference when experts were organized into
clear math and medical groups. Unstructured heterogeneity, not task count by
itself, was the core problem.

### Stronger bases and larger models help, but do not remove the constraints

[What Matters for Model Merging at Scale?](https://arxiv.org/abs/2410.03617)
studied fully fine-tuned PaLM-2 experts from 1B to 64B parameters. Stronger base
models, instruction-tuned bases, and larger models generally retained expert
capabilities better. At the largest scale, differences among several merging
methods narrowed and simple averaging became competitive.

[MergeBench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/91f7f71cb04699f387dc863da42a1fe3-Abstract-Datasets_and_Benchmarks_Track.html)
reaches a compatible conclusion across Llama and Gemma models from 2B to 9B and
five domains: stronger bases, coefficient tuning, and appropriate sparsity help
knowledge retention, but merged models can still trail joint multi-task
training in-domain.

Scale gives the model more capacity and a more stable base. It does not make
arbitrary checkpoints compatible.

## The main technique families

### Checkpoint averaging, EMA, SWA, and LAWA

These methods average checkpoints from one training trajectory. They are the
lowest-risk form of souping because parameter alignment and training history
are shared by construction.

For LLM pretraining, [LAWA](https://arxiv.org/abs/2306.03241) averages separated
recent checkpoints and reports improvements across nanoGPT and Pythia models
from 125M to 12B, especially in early and middle training. The averaging window
still matters: checkpoints from very early incompatible phases should not be
mixed, and gains can diminish late in training.

### Uniform, greedy, and learned soups

- **Uniform soup** is cheap and useful when every ingredient is already strong.
- **Greedy soup** is safer when candidates come from a broad hyperparameter
  sweep.
- **Learned soups** or methods such as
  [LM-Cocktail](https://aclanthology.org/2024.findings-acl.145/) use a small
  calibration set to tune coefficients and often interpolate back toward the
  base to preserve general capability.

Greedy and learned methods replace an assumption with a selection problem. They
are safer, but they can overfit the merge-validation set if the same benchmark
is repeatedly used to select members, tune coefficients, choose the method,
and report the final result.

### Task Arithmetic

[Task Arithmetic](https://openreview.net/forum?id=6t0Kwf8-jrj) treats each
fine-tune delta as a vector that can be added, subtracted, or scaled. It is the
most important baseline for same-base cross-task merging.

Its strengths are simplicity and explicit control over distance from the base.
Its weaknesses are equally direct: the correct base must be known, update norms
must be comparable, and task directions can conflict.

Always compare normalized and unnormalized variants. In heterogeneous pools,
normalization can prevent the total update norm from growing with the number of
models.

### TIES

[TIES-Merging](https://openreview.net/forum?id=xtaX3WyCj1) addresses two common
problems: many fine-tuning updates are small and redundant, and models may
update the same parameter with opposite signs. TIES:

1. trims small-magnitude updates;
2. elects a dominant sign at each parameter position;
3. merges only updates that agree with the elected sign.

This can work well for controlled experts with coherent task roles. It is not a
universal upgrade over Task Arithmetic. In a highly heterogeneous pool, sign
selection can suppress the cancellation that keeps a merge near the base and
retain high-magnitude noise instead.

### DARE

[DARE](https://arxiv.org/abs/2311.03099) is best understood as a delta
preprocessor, not a complete merger. It randomly drops a fraction $p$ of a
fine-tuning delta and rescales the retained entries by $1/(1-p)$, then feeds the
result into averaging, Task Arithmetic, or TIES.

DARE can discard 90%, and sometimes 99%, of very small supervised-fine-tuning
deltas with little degradation. It is far less reliable when the delta is
large, the model underwent continued pretraining, or the wrong base checkpoint
is used. The paper contains an especially sharp failure: computing a code-model
delta from the wrong base can collapse HumanEval and MBPP performance to zero.

Only sparsify the fine-tuning delta. Do not apply DARE-style dropout to the full
model weights.

### Fisher merging and RegMean

Fisher merging weights parameters by an estimate of their importance.
[RegMean](https://openreview.net/forum?id=FCnohuR6AnM) uses activation Gram
matrices to solve a local regression objective for each linear layer.

These methods can approximate source-model functions more directly than naive
weight averaging, but they are not truly data-free. Fisher needs gradient or
Fisher statistics. RegMean needs calibration activations and can require
substantial statistics storage. Their strongest language-model evidence still
comes mainly from BERT, RoBERTa, DeBERTa, and T5 rather than modern 7B+ decoder
LLMs.

### LoRA and adapter merging

LoRA merging looks easy because all adapters share a frozen base. There is a
subtle problem: a low-rank factorization $\Delta W = BA$ is not unique. Two
adapters can encode useful updates in differently oriented low-rank bases.
Elementwise averaging of the $A$ and $B$ factors can therefore be misleading.

The safer procedure is to materialize each actual $\Delta W$ and merge those
updates. [LoraHub](https://arxiv.org/abs/2307.13269) uses a small number of target
examples and gradient-free coefficient search. It improves over zero-shot and
IA3 baselines on BBH, but still trails full fine-tuning and becomes less stable
when the candidate pool grows.

[KnOTS](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d4f8a5109c5083b5307fcd0bddae7a7-Abstract-Conference.html)
uses a joint SVD to transform LoRA updates into a shared aligned space before
applying existing merging rules, improving LoRA merging by up to 4.3% across
its language and vision evaluations.

LoRA composition becomes much harder when adapters change prompt format,
style, refusal behavior, and domain capability simultaneously. At that point,
a router may be better than another static coefficient search.

### SLERP, layer-wise merging, and evolutionary search

SLERP interpolates two weight vectors along a spherical path. It is a useful
engineering candidate for two nearby checkpoints, but it originated in
quaternion graphics, not a general theory that neural-network weights live on
a meaningful sphere. Evidence on LLMs is mixed.

Layer-wise or “Frankenmerge” recipes select or stack Transformer blocks from
different models. The risk is activation-distribution mismatch between adjacent
layers. [Evolutionary Model Merge](https://www.nature.com/articles/s42256-024-00975-8)
shows that black-box search over layer-wise parameters and data-flow paths can
find strong recipes. It also shows that naive hand-built Frankenmerges can
collapse completely. The success comes from search and evaluation, not from
layer splicing by itself.

### Routing and mixture-of-experts

When source experts are genuinely conflicting, preserving them and learning a
router can be more reliable than forcing all behavior into one dense parameter
set. Branch-Train-Merge and Branch-Train-MiX show this pattern: informed
mixtures and routed experts outperform uniform weight averaging, but require
extra parameters, router training, and more complicated serving.

This is not “free soup.” It is often the correct system design.

## The failure modes to test explicitly

### Hard incompatibility

Stop before merging if tensor shapes, tokenizer rows, positional encoding,
normalization, special tokens, output heads, or PEFT injection points cannot be
mapped exactly. A recipe cannot repair an undefined parameter correspondence.

### Different basins despite a shared base

Sharing a pretrained checkpoint is important but not sufficient. BERT studies
have found different fine-tuning seeds in separate linearly disconnected
basins, associated with different out-of-distribution strategies. A model can
be flat yet rely on a brittle heuristic. Flatness is not a substitute for
behavioral evaluation.

### Update-norm mismatch

SFT, DPO, PPO, RLVR, and continued-pretraining checkpoints can have deltas with
very different norms. Without per-model normalization or scaling, the largest
update dominates the merge. Report the relative delta norm per layer and per
source model.

### A bad ingredient

Uniform averaging has no protection against a failed run. Validate every
ingredient independently. Greedy admission should reject a model if it improves
one benchmark while violating a core capability or safety guardrail.

### Safety and alignment regression

Capability scores do not imply preserved alignment. In
[One Bad Model Spoils the Bunch](https://aclanthology.org/2024.findings-emnlp.762/),
seven Mistral ingredients had Llama-Guard-2 alignment rates from 61.8% to 93.0%.
Their public DARE merge scored 53.0%—worse than every source model.

One safety classifier is not a complete safety evaluation. But the result is
enough to reject the assumption that a convex combination automatically
inherits the safest ingredient.

Treat safety, refusal quality, over-refusal, helpfulness, factuality, formatting,
and jailbreak robustness as separate merge objectives and hard deployment
constraints.

## A practical merge protocol

### 1. Audit lineage before touching weights

For every checkpoint, record:

- exact base revision and hash;
- architecture config;
- tokenizer files and vocabulary size;
- special-token IDs and chat template;
- fine-tuning method and training domain;
- number of steps and learning rate, if known;
- continued pretraining versus SFT versus preference/RL training;
- LoRA configuration;
- quantization state;
- license and redistribution obligations.

If the exact base or tokenizer correspondence is unknown, do not begin with a
dense merge.

### 2. Run cheap geometry checks

Compute:

- $\lVert\Delta_i\rVert / \lVert\theta_0\rVert$ and per-layer delta norms;
- pairwise task-vector cosine similarity;
- sign disagreement among top-magnitude updates;
- pairwise interpolation curves for loss, perplexity, core skills, and safety.

For two models, an initial interpolation grid of
$\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$ is often enough to detect a severe
barrier. Use a denser grid when the midpoint looks suspicious.

These statistics are screening signals, not a merge oracle. Near-zero cosine
can mean complementary directions or unrelated noise.

### 3. Establish baselines that cannot be skipped

Evaluate:

1. the base model;
2. every source expert;
3. the best single expert;
4. uniform averaging;
5. normalized Task Arithmetic;
6. interpolation between the proposed merge and the base;
7. an output ensemble as an upper-bound reference, when affordable;
8. joint multi-task or continued training, when source data is available.

If a complex method cannot consistently beat simple averaging or normalized
Task Arithmetic, it has no engineering justification.

### 4. Search conservatively

A reasonable starting grid—not a universal optimum—is:

- Task Arithmetic scale
  $\lambda \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$;
- TIES density in $\{0.05, 0.1, 0.2, 0.5, 1.0\}$, jointly tuned with a
  conservative global scale;
- DARE drop rate in $\{0.5, 0.7, 0.9\}$ only when the base is correct and the
  deltas are small;
- greedy membership where each admission must improve the combined validation
  objective and pass every hard guardrail;
- nonnegative or sum-to-one coefficient constraints when unconstrained search
  drifts too far from the base.

Separate merge calibration, method selection, and final testing. Tuning all
recipes on the reported benchmark creates a multiple-comparison problem, even
when no training examples are used.

### 5. Evaluate the worst task, not only the average

The evaluation matrix should include:

| Axis               | What to measure                                                           |
| ------------------ | ------------------------------------------------------------------------- |
| Expert retention   | Each source model's core task, relative retention, worst-task degradation |
| General capability | Reasoning, knowledge, reading, and base-model regression                  |
| Generation         | Instruction following, output format, tool schema, long context           |
| Distribution shift | Held-out domains and fresh final-only benchmarks                          |
| Safety             | Harmfulness, jailbreaks, over-refusal, and helpfulness                    |
| Language           | Target-language and cross-lingual behavior                                |
| Systems            | VRAM, latency, throughput, context length, and quantization compatibility |
| Stability          | Multiple prompt templates, decoding settings, and seeds where stochastic  |

Before expensive evaluation, run smoke tests for tokenizer behavior, special
tokens, chat formatting, short-text perplexity, repetition, and gibberish.

### 6. Predefine the deployment gate

A production merge should satisfy a rule written before the final test:

- it beats the base and simple-merge baseline on the primary objective;
- every critical expert capability stays above its retention threshold;
- worst-task degradation stays within a fixed budget;
- safety is non-inferior to the production baseline;
- tokenizer, template, long-context, and tool-use behavior do not regress;
- the gain justifies recipe search, evaluation, storage, and governance cost.

“Training-free” never means “evaluation-free.”

## A decision rule by use case

If the goal is **more robust fine-tuning on one task**, start with checkpoint
averaging or greedy soup. Keep the head and tokenizer fixed, reject failed runs,
and expect a small but potentially reliable improvement.

If the goal is **one dense math, code, and medical model**, require a shared
base and begin with normalized Task Arithmetic plus base interpolation. Try
TIES or DARE-TIES only after measuring sign conflict and delta scale. Give every
domain its own guardrail. If the model cannot retain all domains, switch to a
router, MoE, or distillation.

If the goal is **combining popular Hugging Face checkpoints**, treat the
project as high-risk research. Verify lineage and licenses, cluster models by
task and training method, exclude abnormal delta norms and unsafe ingredients,
and set the expectation to “small improvement without damaging the base,” not
“automatic supermodel.”

If the goal is **combining LoRAs**, verify the exact base revision, target
modules, rank, scaling, and tokenizer. Materialize the actual deltas. Two clear
skills may work with coefficient search. Multiple adapters with basis mismatch
may need SVD alignment. Conflicting style, format, and safety adapters may need
a router.

If the goal is **preserving safety or preference alignment**, make safety an
explicit task and a hard constraint. Compare SFT, DPO, RLHF, and RLVR update
norms before merging. Do not assume that adding one aligned model makes the
mixture aligned.

## Final takeaways

1. Model souping is interpolation, not automatic knowledge composition.
2. Same task, same base, and modest fine-tuning distance is the highest-
   reliability regime.
3. A shared base is necessary for most practical dense merging, but it is not
   sufficient; different fine-tunes can still occupy different basins.
4. Ingredient quality matters more than recipe complexity.
5. Task Arithmetic is the essential LLM baseline; TIES, DARE, Fisher, RegMean,
   and subspace methods are conditional tools, not universal upgrades.
6. Larger and stronger bases tend to merge better, but do not make arbitrary
   community checkpoints compatible.
7. LoRA factors are not automatically aligned; merge actual deltas or align
   their subspaces.
8. Keep the merged update close to the base unless validation provides strong
   evidence for moving farther.
9. Evaluate expert retention, general capability, OOD behavior, and safety
   separately.
10. When experts genuinely conflict, routing or distillation may be a better
    architecture than a single dense soup.

The practical standard should therefore be modest: use model souping as a
low-cost composition tool under strong compatibility conditions. In
heterogeneous settings, treat evaluation, routing, and retraining as first-class
alternatives—not fallback ideas after a leaderboard merge fails.
