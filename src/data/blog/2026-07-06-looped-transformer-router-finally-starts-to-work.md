---
author: Jing Lu
pubDatetime: 2026-07-06T18:00:00Z
title: "A Looped Transformer Router Shows Its First Replicated Gain"
featured: false
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Pretraining
  - Scaling Laws
description: "A small-budget BPE language-model experiment where a sparse late-final-loop token-feedback router becomes the first route-looped Transformer candidate to beat matched fixed-loop baselines across several controlled checks."
---

The useful result from this week is small, but real enough to change the research direction.

After many routed looped Transformer variants that were either unstable, collapsed, or only improved one metric while hurting another, we finally have a replicated router candidate that beats a matched fixed loop baseline on a small BPE language-model setup.

This follows an earlier negative result where a sequence-level router looked competitive but collapsed toward a low-entropy exit policy. The new result is different because the router is token-conditioned, sparse, and only active late in the recurrent computation.

The July 10 update is that I am now comfortable calling this a **candidate**, not only a promising run. It has cleared the same matched-fixed gate across multiple controlled checks: d384, d512, matched-depth `4x2`, depth-16 `4x4`, and a batch-size-8 confirmation. The margins are still modest, but the sign is finally consistent.

The current best candidate is:

```text
d512 / effective depth 16 / fixed_4x4 baseline
vs.
late-final-loop 4x4 token-feedback router

batch size = 8
steps = 2000
seeds = 0, 1, 2
tokenizer = GPT-2 BPE
training data = nanochat-style text
train cap = 8 MB
eval cap = 512 KB
```

This is not a solved architecture yet. The margins are still tiny, around `1e-3` in broad language-model loss. But the important change is that the router no longer wins only by sacrificing the general language-model objective. It is slightly better on validation loss, token accuracy, reasoning-slice loss, deterministic validation loss, easy-token loss, and fixed-defined hard-token loss at the same time.

The `8 MB` train cap is intentional. This is a screening setup for architecture decisions, not evidence of pretraining-scale behavior.

The short version:

```text
The first router candidate that looks worth scaling is not an early, broad,
free-form router. It is a sparse late-final-loop feedback router.
```

## The Core Idea

A normal Transformer spends depth in a straight line:

```text
block 1 -> block 2 -> block 3 -> block 4
```

A looped Transformer reuses a smaller number of blocks multiple times. A `4x4` loop means:

```text
4 unique recurrent blocks
4 recurrent passes
effective depth = 16 block applications
```

That already gives a useful parameter-sharing structure: fewer unique blocks, more effective computation. It does not automatically make inference cheaper, because recurrent visits still have to be executed.

The routed version asks for something stronger. Instead of forcing every token and every context through the same fixed recurrent path, can the model learn when later block state should feed back into earlier computation?

The research question is:

```text
Can later block outputs feed back into earlier blocks, and can a router decide
when that feedback is useful based on token and context state?
```

## Why Earlier Routers Failed

The negative results were the most useful part of the project.

The first sequence-level destination routers could sometimes beat fixed loops at medium training budgets, but the route policy often collapsed. Route entropy fell, final-exit probability rose toward one, and the router became more like a cheap exit policy than an adaptive computation policy.

Then token-level destination routers gave a better signal on hard tokens, but they often paid for it elsewhere. Some variants improved reasoning loss or fixed-defined hard-token loss while hurting generic validation loss. That made them interesting diagnostics, not architecture candidates.

The lesson was:

```text
The router has signal, but the objective must protect generic likelihood while
letting the model spend a small amount of feedback on genuinely useful tokens.
```

Only winning on hard tokens is not enough. A candidate has to beat the fixed loop on the full bundle:

| Gate | Why it matters |
|---|---|
| Validation loss | General language-model quality |
| Token accuracy | Direct next-token performance |
| Reasoning-slice loss | A proxy for multi-step language-like examples |
| Deterministic validation loss | Less noisy paired comparison |
| Easy-token loss | Checks that routing does not damage obvious tokens |
| Hard-token loss | Checks whether extra computation helps difficult tokens |

## The Candidate That Worked

The best current router is not a free-form destination router. It is a late token-feedback router.

Instead of choosing an entirely different path at every step, it adds a small learned feedback signal late in the recurrent computation:

```text
later block state
  -> token/context router
  -> learned source feedback
  -> add a small correction to the current hidden state
```

The strongest structure so far is:

```text
fixed_4x4 backbone
4 unique recurrent blocks
4 recurrent passes
effective depth = 16

router active only in the final recurrent loop
router active only on late blocks 3 and 4
token_feedback_applications = 2
```

In plain language: let the fixed recurrent computation build a stable representation first, then allow a small amount of late feedback for tokens that appear to benefit from it.

That constraint matters. Earlier variants that applied feedback more broadly were noisier. The router needs to be useful, but also quiet.

## Why I Am Calling It a Candidate

The promotion criterion was not "one run beats fixed." The router had to beat the matched fixed-loop baseline on the full metric bundle across larger checks.

The current candidate family is:

```text
token_feedback_router_latefin4x4_learnedsrc_scale010_dynfloor000_gate005_utilbce_w005_hard030_negbce004_top010_protectbce002_top010_pos015_start300_biasm4
```

For the matched-depth `4x2` checks, the sibling candidate is the same learned-source feedback router restricted to late blocks:

```text
token_feedback_router_lateonly4x2_learnedsrc_scale010_dynfloor000_gate005_utilbce_w005_hard030_negbce004_top010_protectbce002_top010_pos015_start300_biasm4
```

Here is the evidence stack that changed my mind:

| Check | Matched baseline | Seeds | Result |
|---|---|---:|---|
| d384, `4x2`, batch4, 2000 steps | `fixed_4x2` | 0/1/2 | Router wins token accuracy, validation loss, reasoning loss, deterministic validation loss, easy-token loss, and hard-token loss |
| d512, `4x2`, batch4, 2000 steps | `fixed_4x2` | 0/1/2 | Same all-metric win at larger width |
| d512, depth16 `4x4`, batch4, 2000 steps | `fixed_4x4` | 0/1/2 | Same all-metric win after moving to the final-loop-only depth-16 structure |
| d512, depth16 `4x4`, batch8, 2000 steps | `fixed_4x4` | 0/1/2 | Strongest current confirmation; hard-token margin widens |

The important part is not that any one metric is dramatic. It is that the sign does not flip when the comparison gets stricter. Earlier routers often had one beautiful column and one ugly column. This one is boring in the right way: small positive margins across the whole gate.

## Strongest Three-Seed Result

The exact promoted candidate is:

```text
token_feedback_router_latefin4x4_learnedsrc_scale010_dynfloor000_gate005_utilbce_w005_hard030_negbce004_top010_protectbce002_top010_pos015_start300_biasm4
```

Matched baseline:

```text
fixed_4x4
```

Three-seed aggregate:

| Metric | `fixed_4x4` | router | router - fixed |
|---|---:|---:|---:|
| Validation token accuracy | `0.883870` | `0.883952` | `+0.000081` |
| Last validation loss | `1.338885` | `1.337927` | `-0.000958` |
| Reasoning loss | `1.237956` | `1.237297` | `-0.000659` |
| Deterministic validation loss | `1.336670` | `1.335787` | `-0.000883` |
| Easy-token loss | `0.031210` | `0.031138` | `-0.000071` |
| Hard-token loss | `5.253051` | `5.249732` | `-0.003318` |

Lower loss is better. Higher accuracy is better.

The router statistics are also important:

| Router statistic | Value |
|---|---:|
| Mean token feedback | `0.057688` |
| Source entropy | `0.366058` |
| Source final mass | `0.856165` |
| Feedback applications | `2.0` |

The router is mostly using final-ish source information, but it has not collapsed into a completely deterministic route. It learns a small source mixture and applies feedback to only a small share of token states.

That is exactly the behavior we wanted to see before scaling the experiment further.

This is a matched-backbone training-quality comparison, not a serving-cost claim. The fixed and routed models use the same `4x4` recurrent backbone and the same training setup, but the router adds extra routing machinery and auxiliary losses. A separate serving experiment would be needed to show an actual latency or FLOP advantage.

## Why Batch Size Mattered

The same depth-16 direction was already slightly positive at batch size `4`, but the margins were almost too small to trust:

| Metric | batch4 router - fixed |
|---|---:|
| Validation loss | `-0.000430` |
| Reasoning loss | `-0.000450` |
| Deterministic validation loss | `-0.000438` |
| Hard-token loss | `-0.000986` |

At batch size `8`, the same broad pattern became clearer:

| Metric | batch8 router - fixed |
|---|---:|
| Validation loss | `-0.000958` |
| Reasoning loss | `-0.000659` |
| Deterministic validation loss | `-0.000883` |
| Hard-token loss | `-0.003318` |

The most interesting movement is hard-token loss: the advantage widened from about `0.0010` to about `0.0033`.

My current hypothesis is that router training is more sensitive to gradient noise than the fixed baseline. The fixed model only has to learn the language-model objective. The router has to learn the language-model objective, identify useful token states, avoid damaging easy tokens, and keep the feedback policy sparse. A small batch can make those auxiliary signals too noisy.

## What This Does Not Prove

This result should be framed carefully.

It does not prove that routed looped Transformers are generally better than fixed looped Transformers. It does not prove a scaling law. It does not yet prove that routing will help at larger natural-language pretraining scale. It also does not claim a compute-efficiency win at inference time.

It also does not mean "deep thinking is solved." The reasoning slice here is a useful language-model diagnostic, not a proof benchmark. The result says that late feedback can improve a matched next-token setup without paying a broad validation tax. That is a much narrower and more useful claim.

Finally, this is separate from the original synthetic minimization target. The earlier synthetic track used a `gain_preserved >= 0.90` acceptance gate for preserving full-loop gains with a smaller structure. This blog post is about the language-model router track: can a routed feedback structure beat a matched fixed recurrent loop in small-budget BPE pretraining? Those are related research questions, but they should not be mixed.

What it does prove is narrower:

```text
A late, sparse, token-feedback router can beat a matched fixed_4x4 looped
Transformer baseline in a replicated small-budget BPE language-model run.
```

That is enough to promote the design from "diagnostic experiment" to "current router candidate."

## Source Artifacts

The promoted result is tracked in the project-local candidate note:

```text
docs/router_candidate_20260709.md
```

The project registry is:

```text
docs/current_candidate.json
```

I also kept a compact project-local summary for this exact batch8 result:

```text
docs/batch8_latefin4x4_evidence_20260706.md
```

The two Modal summaries behind the batch8 three-seed aggregate are:

```text
runs/modal-downloads/nanochat_lm_bpe_feedback_latefin4x4_learnedsrc_negbce004_d512_eval4_b8_2000s_seed0_20260706/summary.json
runs/modal-downloads/nanochat_lm_bpe_feedback_latefin4x4_learnedsrc_negbce004_d512_eval4_b8_2000s_seeds12_20260706/summary.json
```

The earlier batch4 depth-16 aggregate is also kept in the same registry, so the batch4-to-batch8 comparison is not a separate hand calculation.

## The Research Direction From Here

The next experiment should scale this exact structure before inventing a more complex router. I do not want to spend GPU budget confirming old variants that already lost.

The decision gate should be:

1. Keep the fixed `4x4` matched baseline.
2. Scale model width or depth while preserving the late-final-loop feedback pattern.
3. Keep three-seed comparisons.
4. Require the router to win validation loss and reasoning loss, not only hard-token loss.
5. Track route statistics so a win caused by collapse does not get promoted.

Because the current margins are still small, the scale-up path should be conservative:

1. Re-run the current best d512 depth-16 batch8 shape with one larger budget axis at a time.
2. Start with a single seed scout before spending on seeds 1 and 2.
3. Stop early if the router loses broad validation loss or hard-token loss.
4. Only after that consider a wider model.

The goal of the next run is not another tiny positive result. The goal is to see whether the margin widens when the router gets a more stable training signal.

The architectural lesson is also becoming clearer:

```text
Do not let the router rewrite the computation path too early.
Let recurrence build stable state first.
Then use a small, late, token-conditioned feedback correction.
```

That may be the minimal structure that makes routed looped Transformers worth scaling.
