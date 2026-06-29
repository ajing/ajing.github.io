---
author: Jing Lu
pubDatetime: 2026-06-29T10:00:00Z
title: "How to Test Pretraining Ideas at Small Scale Before Betting on a Large Model"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Pretraining
  - Scaling Laws
description: "A practical guide to validating pretraining improvements with small proxy models, scaling ladders, isoFLOP budgets, loss curves, downstream evals, and rank-correlation checks before committing to an expensive large-model run."
---

Large-model pretraining is too expensive to run by intuition. If you have a new data filter, tokenizer, deduplication rule, domain mix, curriculum, optimizer setting, or architecture tweak, you cannot simply train a 70B model and hope the gain survives.

The normal practice is to build a scaling ladder.

You test the idea on small models. Then you test it on medium models. Then you check whether the gain behaves predictably as model size, token count, and compute increase. Only after the improvement survives several controlled scale points do you spend the money on the large run.

This post explains how that is usually done in pretraining practice, and why the method works.

The core idea:

> A pretraining improvement is not real just because it improves a 100M model.
>
> It becomes credible when the improvement is stable across model sizes, token budgets, seeds, validation slices, and downstream evaluations.

## 1. Why small models are useful at all

The reason small models are useful is not that they are miniature copies of large models in every way. They are not.

Small models are useful because many pretraining metrics scale smoothly. Kaplan et al. showed that language-model cross-entropy loss follows approximate power-law relationships with model size, dataset size, and compute over a wide range of scales ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)). Hoffmann et al. later showed that compute-optimal training depends strongly on the balance between model parameters and training tokens, not just parameter count ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)).

This gives pretraining teams a way to ask:

- If this data change improves loss at 100M, does it still improve loss at 300M?
- If it improves loss at 300M, does it still improve at 1B?
- Is the improvement shrinking, flat, or growing with scale?
- Does it improve only validation loss, or also downstream tasks?
- Does it improve only easy evals, or also tasks we actually care about?
- Does it help only under one token budget, or across multiple budgets?

The goal is not to prove the future perfectly. The goal is to reduce the chance that a large run surprises you.

## 2. The basic variables: N, D, C, and Q

A practical scaling experiment usually tracks four variables.

**N: model size.** This is usually the number of non-embedding or total trainable parameters: 70M, 160M, 410M, 1B, 3B, 7B, and so on.

**D: training tokens.** This is how many tokens the model sees. In Chinchilla-style compute-optimal training, model size and token count scale together. Hoffmann et al. trained more than 400 models from 70M to over 16B parameters and argued that, under a fixed compute budget, parameters and tokens should scale roughly together ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)).

**C: compute.** A rough rule of thumb for dense decoder-only Transformer training is that training FLOPs scale like:

```text
C ~= 6 * N * D
```

This approximation is not exact, but it is useful for comparing experiment budgets.

**Q: data quality or recipe quality.** This is the part you are testing: a new filter, a new data source, a new mixing ratio, a new deduplication policy, a new tokenizer, a new schedule, or a new architecture detail.

Scaling-law work usually models loss as a smooth function of model size, data size, and compute. In practice, pretraining teams often care about whether changing Q shifts the curve downward.

If your intervention lowers the loss curve at every scale, it is promising.

If it lowers loss only for tiny models and disappears at 1B, it is probably a small-model artifact.

If it helps loss but hurts downstream tasks, the intervention may be optimizing the wrong proxy.

## 3. The key experimental question

The question is not:

> Did my small model get better?

The question is:

> Does my small-model result predict a better large-model run?

Those are different.

A 100M model may prefer cleaner, simpler, more textbook-like data because it has limited capacity. A 7B model may benefit from broader, messier, more diverse web data because it can model more variation. A small model may be bottlenecked by optimization instability. A large model may be bottlenecked by data diversity. A small model may underuse code data. A larger model may turn code data into better reasoning and tool-use transfer.

So the job of a scaling experiment is not to crown a winner at the smallest scale. It is to identify which interventions have stable scaling behavior.

## 4. The standard pretraining ladder

A practical ladder has several stages.

### Stage 0: Data and pipeline smoke test

Before training meaningful models, train a tiny model to catch obvious breakage.

Example:

- 10M to 50M parameters;
- 1B to 5B tokens;
- one or two seeds;
- short context length if needed;
- aggressive logging.

The goal is not scientific evidence. The goal is to find broken tokenization, duplicate explosions, bad loss spikes, source corruption, data-loader bugs, unstable gradients, and accidental evaluation leakage.

If a recipe cannot survive Stage 0, it is not ready for real comparison.

### Stage 1: Cheap proxy models

Next, run a small but real comparison.

Example:

- 100M to 300M parameters;
- 10B to 30B tokens;
- baseline recipe vs. candidate recipe;
- identical architecture, tokenizer, optimizer, schedule, context length, and evaluation harness;
- at least two seeds if the gain is small.

This stage tells you whether the intervention has any signal. It is where you test many ideas quickly.

Typical candidates:

- data filter thresholds;
- source-level exclusions;
- quality classifier cutoffs;
- deduplication aggressiveness;
- code/math/web/book mixing ratios;
- tokenizer variants;
- document packing rules;
- curriculum choices;
- learning-rate or batch-size changes.

Most ideas should die here. That is the point.

### Stage 2: Medium proxy models

The surviving ideas move to a medium scale.

Example:

- 400M to 1B parameters;
- 30B to 150B tokens;
- fewer candidates;
- more downstream evaluations;
- stronger contamination checks;
- per-domain validation loss tracking.

This is the first scale where you ask whether the gain is still there under a more realistic model and token budget.

DCLM is a good public example of this mindset. It defines multiple competition scales, including roughly 400M, 1B, and 7B settings, so teams can test data curation methods under smaller compute budgets before checking whether they transfer to larger runs ([DCLM](https://arxiv.org/html/2406.11794v1)).

### Stage 3: Large proxy or pre-final model

Before the final expensive run, train one larger proxy.

Example:

- 3B to 7B parameters;
- 100B to 1T tokens, depending on budget;
- the top one or two candidate recipes;
- full downstream suite;
- full decontamination report;
- early scaling-law fit against the smaller runs.

This is the last chance to catch a false small-scale win.

The large proxy does not need to be the final model size. Its purpose is to answer: does the intervention still help when the model has enough capacity to behave more like the final system?

### Stage 4: Final run

Only now do you train the expensive model.

At this stage, you should not still be debating the basic data filter. You should already know:

- expected validation loss range;
- expected downstream score range;
- expected training stability;
- likely failure modes;
- which evals are noisy;
- which gains are robust;
- how the final run compares to the fitted scaling curve.

The final run is still risky, but it should not be a blind bet.

## 5. Two budget styles: isoFLOP and fixed-token

There are two common ways to compare pretraining interventions.

### IsoFLOP comparison

In an isoFLOP comparison, every candidate gets the same compute budget.

Example:

```text
Baseline: 300M model, 30B tokens
Variant:  300M model, 30B tokens
```

or:

```text
Baseline: 1B model, 50B tokens
Variant:  1B model, 50B tokens
```

This is the cleanest test if you are asking:

> Which recipe gives better performance for the same training cost?

IsoFLOP comparisons are especially important when the candidate changes data quality, data order, optimizer, deduplication, or filtering.

### Fixed-token comparison

In a fixed-token comparison, every candidate sees the same number of tokens.

This is useful if you are asking:

> Does this data distribution produce better learning from the same amount of text?

But fixed-token comparisons can be misleading across model sizes because a larger model trained on the same number of tokens may be undertrained or overtrained relative to another scale.

In practice, good studies use both:

- fixed-token runs to isolate data effects;
- isoFLOP runs to evaluate training-budget efficiency;
- Chinchilla-style or overtraining-aware schedules to pick realistic final budgets.

## 6. What to measure at every scale

You need more than one metric.

### Training loss

Training loss catches optimization problems and gross data difficulty differences. But lower training loss alone can mean the data is easier, more duplicated, or less diverse.

A recipe that lowers training loss by making the corpus repetitive may not produce a better model.

### Validation loss

Validation loss is the primary scaling-law metric. It should be measured on held-out data that is not part of the training corpus.

Use multiple validation slices:

- general web;
- books;
- code;
- math;
- academic text;
- multilingual text;
- instruction-like text;
- high-quality curated text;
- noisy web text;
- domain-specific targets.

The question is not only whether average validation loss improves. The question is where it improves and where it regresses.

### Downstream evaluations

Downstream evals are noisier than validation loss, but they are closer to what users care about.

Evaluate on:

- knowledge QA;
- math;
- code;
- reasoning;
- multilingual tasks;
- long-context tasks if relevant;
- safety or refusal behavior if post-training will depend on it;
- internal target tasks.

Gadre et al. explicitly address a gap in older scaling work: scaling laws often predict next-token loss in compute-optimal regimes, while real models are often overtrained and judged on downstream tasks. They train 104 models from 0.011B to 6.9B parameters and show how cheaper experiments can predict both validation loss and downstream error in more realistic regimes ([Gadre et al., 2024](https://arxiv.org/abs/2403.08540)).

### Per-source and per-domain loss

If the intervention changes data, track per-source loss.

Example:

```text
General web validation loss:    -0.03
Code validation loss:           +0.02
Math validation loss:           -0.06
Academic validation loss:       -0.01
Multilingual validation loss:   +0.04
```

This tells you whether the gain is broad or whether you are trading away one capability for another.

### Memorization and contamination checks

A small model may not memorize a leaked benchmark item, while a larger model can. Contamination risk can therefore increase with scale. So the scaling ladder should include decontamination checks, not just performance checks.

For each candidate recipe, keep track of:

- benchmark overlap;
- duplicated training documents;
- exact and near-duplicate eval matches;
- source-level benchmark mirrors;
- performance gap between suspicious and clean eval subsets.

## 7. The theory: why scaling extrapolation works

The theory is empirical but strong enough to guide engineering.

Kaplan et al. observed that language-model loss can be approximated by power laws in model size, data size, and compute ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)).

A simplified version looks like:

```text
L(N, D) = L_inf + A / N^alpha + B / D^beta
```

Where:

- `L(N, D)` is validation loss;
- `N` is model parameters;
- `D` is training tokens;
- `L_inf` is irreducible loss;
- `A / N^alpha` is the penalty from limited model capacity;
- `B / D^beta` is the penalty from limited data.

Hoffmann et al. refined the practical conclusion: many large models were undertrained on tokens, and compute-optimal training should scale model size and token count together ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)).

For pretraining practice, the key point is not the exact exponent. The key point is that curves are often smooth enough that small experiments can predict larger experiments.

But this only works if the experiment is controlled:

- same architecture family;
- same optimizer;
- same tokenizer or intentionally tested tokenizer;
- same data pipeline except the intervention;
- same evaluation harness;
- comparable token budgets;
- enough scale points to fit a trend;
- enough eval slices to catch regressions.

If you change five things at once, the scaling curve cannot tell you which thing mattered.

## 8. What it means for a gain to "survive scaling"

A gain survives scaling when it passes several checks.

### Check 1: Same sign across scales

If the candidate improves validation loss at 100M, 300M, 1B, and 3B, that is much more credible than a one-off win.

The gain does not have to be identical. It can shrink or grow. But the sign should not randomly flip.

### Check 2: Stable rank ordering

If you compare several candidate data recipes, the ranking should be similar across scales.

DCLM reports high rank correlation between smaller-scale results and larger 7B-scale results, which supports the idea that data curation can be iterated at small scales before larger confirmation ([DCLM](https://arxiv.org/html/2406.11794v1)).

The practical version is:

```text
At 300M:
Recipe C > Recipe A > Recipe B

At 1B:
Recipe C > Recipe A > Recipe B

At 7B:
Recipe C > Recipe A > Recipe B
```

This is stronger evidence than "Recipe C won one small run."

### Check 3: Loss gain maps to downstream gain

Validation loss is smoother than downstream evals, but downstream evals matter. A recipe that improves loss but not downstream performance may still be useful, but it should be treated carefully.

Gadre et al. model the relationship between perplexity and downstream task error, which is valuable because it connects the cheap, smooth metric to the expensive, noisy metric ([Gadre et al., 2024](https://arxiv.org/abs/2403.08540)).

### Check 4: No hidden regressions

A gain is not a gain if it quietly breaks another capability.

For example:

- filtering noisy web improves average loss but hurts multilingual coverage;
- adding code improves coding but hurts natural-language QA;
- adding synthetic math improves math but increases benchmark-style contamination risk;
- aggressive deduplication improves stability but removes rare-domain knowledge;
- quality filtering improves short-context evals but hurts long-tail factual recall.

The scaling ladder should expose these tradeoffs early.

### Check 5: The effect size beats run noise

Small gains can be real, but they are easy to confuse with seed noise, data-order noise, checkpoint selection noise, or eval noise.

If a change improves validation loss by 0.1%, you need more replication than if it improves by 3%.

At minimum:

- repeat small runs with multiple seeds;
- compare checkpoints at the same token count;
- use confidence intervals for downstream evals;
- inspect whether the improvement appears throughout training or only at the final checkpoint.

## 9. The pretraining experiment matrix

A useful experiment matrix looks like this:

| Scale | Params | Tokens | Purpose | Candidate count |
|---|---:|---:|---|---:|
| Smoke | 10M-50M | 1B-5B | Catch pipeline breakage | Many |
| Small proxy | 100M-300M | 10B-30B | Fast recipe search | 10-50 |
| Medium proxy | 400M-1B | 30B-150B | Confirm scale transfer | 3-10 |
| Large proxy | 3B-7B | 100B-1T | Pre-final confirmation | 1-3 |
| Final | 7B+ | target budget | Expensive production run | 1 |

The exact sizes depend on budget. The important part is not the specific numbers. It is the shape:

> Many cheap experiments, fewer medium experiments, one or two large confirmations.

This is how you avoid spending final-run compute on ideas that were never properly derisked.

## 10. How to test a data-filtering idea

Suppose you have a new quality filter for web data.

Bad experiment:

```text
Train one 100M model on filtered data.
Compare it to an old baseline.
Declare victory.
```

Better experiment:

```text
Baseline data: current web mixture
Variant A: mild filter
Variant B: medium filter
Variant C: aggressive filter

Run 100M models for 20B tokens.
Run 300M models for 30B tokens.
Run 1B models for 100B tokens for the top two variants.
Track validation loss by domain.
Track downstream evals.
Track document diversity and deduplication rate.
Track contamination risk.
Fit scaling curves.
Pick the variant whose gains survive and whose regressions are acceptable.
```

The most important question is not "which filter makes the data cleanest?" It is:

> Which filter produces the best model at the final scale and budget?

Those can differ.

Sorscher et al. study data pruning from a scaling-law perspective and show that high-quality data selection can beat naive power-law scaling in some regimes, but the benefit depends on having a good pruning metric ([Sorscher et al., 2022](https://arxiv.org/abs/2206.14486)). This is exactly why data filters need scale validation.

## 11. How to test a data-mixture idea

Now suppose you want to change the mixture:

```text
Current:
70% web
10% books
10% code
5% academic
5% math

Candidate:
55% web
10% books
20% code
5% academic
10% math
```

This is harder than testing a simple filter because the gain may be task-specific.

You need to ask:

- Does code loss improve?
- Does natural-language loss regress?
- Does math improve because of math data or because code teaches structure?
- Does the model become worse at multilingual text?
- Does the downstream suite overweight the domains you added?
- Does the target product actually need the improved domain?

The best practice is to evaluate the mixture both globally and by slice. If a mixture improves the headline average but hurts your target domain, it is not the right mixture.

Recent work on data ablations and mixture scaling tries to make this cheaper. For example, scalable data-ablation methods approximate how subsets of a corpus contribute without training every possible mixture from scratch ([Scalable Data Ablation Approximations](https://arxiv.org/html/2410.15661v1)).

## 12. How to test a tokenizer idea

Tokenizer changes are dangerous because they alter almost everything:

- sequence length;
- token frequency;
- effective context capacity;
- compression by language;
- code representation;
- rare-word behavior;
- optimizer dynamics.

A tokenizer that improves English web loss may hurt code or multilingual text. A tokenizer that reduces token count may make training look cheaper while changing the actual amount of information processed.

For tokenizer experiments, compare:

- bits per byte or byte-normalized loss, not only token loss;
- downstream evals by language and domain;
- code and math formatting behavior;
- average sequence length per source;
- throughput and memory;
- final quality per FLOP, not just per token.

Do not scale up a tokenizer change from one tiny run. Tokenizers are too entangled with the whole training system.

## 13. How to test an architecture or optimizer idea

Architecture and optimizer changes are more fragile than data changes.

A data filter may transfer across scales because it changes the distribution the model learns from. An optimizer change may work at 300M and fail at 7B because stability, batch size, gradient noise, and hardware details change.

For architecture or optimizer tests:

- run more seeds;
- test multiple learning rates;
- test at more than one batch size;
- watch loss spikes;
- compare wall-clock throughput;
- compare memory;
- test long enough to observe the stable regime;
- avoid mixing architecture changes with data changes.

The final metric should be quality per unit cost, not loss alone.

## 14. Why downstream gains can lag behind loss gains

Sometimes validation loss improves before downstream evals improve. This can happen because downstream tasks are noisy, sparse, or thresholded.

Example:

```text
Validation loss improves smoothly:
100M: -0.020
300M: -0.025
1B:   -0.030

Downstream score:
100M: no change
300M: +0.3
1B:   +1.2
```

This is not necessarily a failure. It may mean the capability only becomes visible once the model has enough capacity.

The reverse can also happen:

```text
Validation loss: tiny change
Code eval: large gain
```

That may happen when the intervention improves a narrow domain that is diluted in average validation loss.

This is why you need both smooth aggregate metrics and targeted evals.

## 15. Why gains sometimes disappear at larger scale

Small-scale gains disappear for several reasons.

### The small model was capacity-limited

Cleaner or simpler data may help a small model because it cannot represent the full data distribution. A larger model may no longer need that simplification.

### The metric was too narrow

A small eval suite may reward a narrow domain. A larger eval suite reveals regressions.

### The token budget changed the answer

A filter may help at 20B tokens but hurt at 1T tokens because diversity becomes more important later.

### The intervention changed optimization, not final capability

Some changes make early training faster but do not improve final loss.

### The candidate overfit the proxy eval

If you tune repeatedly on the same small eval suite, the recipe can overfit the development benchmark.

### The final model is overtrained

Many production models are trained on more tokens than compute-optimal Chinchilla prescriptions because inference cost matters. Gadre et al. study this overtraining regime directly and show why extrapolation should account for both model size and token count ([Gadre et al., 2024](https://arxiv.org/abs/2403.08540)).

## 16. The role of open scaling suites

Open model suites are useful because they show what controlled scaling looks like.

Pythia trained 16 models from 70M to 12B parameters on public data in the same order, with many checkpoints released throughout training ([Pythia](https://arxiv.org/abs/2304.01373)). This makes it possible to study how behavior changes with both scale and training time.

OLMo and Dolma are useful because they expose the training data, model checkpoints, and pretraining recipe more openly than most frontier systems ([OLMo](https://arxiv.org/html/2402.00838v1); [Dolma](https://allenai.github.io/dolma/)).

DCLM is useful because it turns data curation into a controlled benchmark: fixed model recipes, multiple compute scales, and a focus on whether data decisions transfer from small to larger models ([DCLM](https://arxiv.org/html/2406.11794v1)).

Together, these projects show the practical ideal:

> Do not just release a final model. Release enough of the training ladder that people can understand why the final model was expected to work.

## 17. A decision rule before scaling up

Before moving from small proxy to medium proxy, require:

- candidate beats baseline on average validation loss;
- no severe per-domain regression;
- training is stable;
- contamination risk is not worse;
- improvement appears before the final checkpoint;
- at least one downstream family improves or remains neutral.

Before moving from medium proxy to large proxy, require:

- same sign of validation-loss gain across two or more scales;
- rank ordering is stable if comparing multiple recipes;
- downstream evals are directionally consistent;
- effect size is larger than run noise;
- no unacceptable target-domain regression;
- cost impact is understood.

Before final training, require:

- a fitted scaling estimate;
- confidence interval or uncertainty range;
- expected final loss and eval range;
- a rollback plan if early final-run loss deviates from prediction;
- decontamination report;
- data provenance report;
- monitoring for training instability.

This sounds bureaucratic, but it is cheaper than wasting a final run.

## 18. What the final scaling report should contain

A good pretraining scaling report should include:

### Experiment table

```text
Run ID
Model size
Token budget
Compute budget
Data mixture
Filter version
Tokenizer version
Optimizer settings
Seed
Hardware
Checkpoint schedule
```

### Loss curves

Show training and validation loss against:

- tokens;
- FLOPs;
- wall-clock time;
- model size;
- data source.

### Scaling fit

Show the fitted curve and where each run lands.

At minimum:

```text
Baseline predicted final loss:  X
Candidate predicted final loss: Y
Expected gain:                 X - Y
Uncertainty:                   +/- Z
```

### Downstream eval table

Separate:

- broad average;
- target-domain average;
- math;
- code;
- knowledge;
- multilingual;
- long-context;
- safety or policy-relevant behavior;
- contaminated vs. clean subsets where relevant.

### Regression table

List the things that got worse. This is where many weak scaling reports fail. They only show wins.

### Go / no-go decision

End with a decision:

- scale up;
- run one more proxy;
- reject;
- accept only for a target-domain model;
- keep for post-training instead of pretraining.

## 19. A concrete example

Suppose you want to add more math and code to pretraining because you believe it will improve reasoning.

The small-scale plan:

```text
Baseline: current mixture
Variant A: +5% code, +5% math, -10% web
Variant B: +10% code, +10% math, -20% web
Variant C: +15% code, +15% math, -30% web
```

Run:

```text
100M for 20B tokens
300M for 30B tokens
1B for 100B tokens on the top two variants
```

Measure:

```text
General validation loss
Code validation loss
Math validation loss
Web validation loss
Multilingual validation loss
MMLU-like evals with contamination checks
GSM8K-style math evals
HumanEval-style code evals
Internal target tasks
```

Decision:

- If Variant C wins math/code but badly hurts multilingual and general web, it may be too aggressive.
- If Variant A improves code/math without regressions, it may be the safest general-model choice.
- If Variant B has the best scaling trend at 1B and no target-domain regressions, send it to the 3B or 7B proxy.

The important point is that you are not choosing based on a single score. You are choosing based on a scaling pattern.

## 20. The mental model

Pretraining is an expensive search problem under uncertainty.

Small models are probes. Medium models are confirmation. Large proxies are rehearsal. The final model is the expensive bet.

Scaling laws are the reason this is possible. They do not remove uncertainty, and they do not guarantee every small-scale improvement will transfer. But they let you replace guesswork with a disciplined sequence:

```text
small experiment
-> controlled comparison
-> scaling curve
-> downstream validation
-> regression audit
-> large-run decision
```

The best pretraining teams are not the ones that simply train the biggest model. They are the ones that can predict, before the final run, which recipe deserves the compute.

## References

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/html/2406.11794v1)
- [Language models scale reliably with over-training and on downstream tasks](https://arxiv.org/abs/2403.08540)
- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/html/2402.00838v1)
- [Dolma: an open corpus of three trillion tokens for language model pretraining research](https://allenai.github.io/dolma/)
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
- [FineWeb: decanting the web for the finest text data at scale](https://arxiv.org/html/2406.17557v1)
- [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486)
- [Scalable Data Ablation Approximations for Language Models](https://arxiv.org/html/2410.15661v1)
