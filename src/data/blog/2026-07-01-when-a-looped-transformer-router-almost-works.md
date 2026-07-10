---
author: Jing Lu
pubDatetime: 2026-07-01T18:30:00Z
title: "When a Looped Transformer Router Almost Works"
featured: false
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Pretraining
  - Scaling Laws
description: "A controlled small-scale language-composition experiment comparing fixed and routed looped Transformers, showing why the first sequence-level router was competitive but collapsed toward a weak exit policy."
---

We tested a simple routed looped Transformer against a fixed looped baseline on a synthetic language-like next-token task. The result is useful, but not in the way we hoped.

This is the negative/control result that came before the later token-feedback candidate. I am publishing it because the numbers explain why the successful direction had to change: the first router could win medium-budget points, but it did not learn a durable adaptive-computation policy.

The router is competitive, and after adding a second interleaved grid it looks slightly more promising than the first pass. Across the controlled grid it wins 17 out of 27 seed-level fixed-vs-routed comparisons, with a small mean loss advantage of `-0.00224`. But the advantage is still fragile. At the longest `1600`-step horizon, it wins only 1 out of 3 new comparisons, and two smaller widths reverse against the fixed loop.

The diagnosis is that the current router learned a weak policy: mostly route toward final exit. In the new `d_model=80`, `1600`-step run, the final exit mass reached `0.9914` and route entropy fell to `0.0127`. That is not a robust adaptive computation policy. It is closer to a cheap-exit baseline.

The useful conclusion is narrow:

```text
Synthetic language-like next-token data is a good bridge task for looped
Transformer research, but the current sequence-level router is only a weak
router candidate until we fix route collapse.
```

The follow-up result is now clear: the next credible candidate was not another sequence-level destination router. It was a sparse late-final-loop token-feedback router, described in the later post [A Looped Transformer Router Shows Its First Replicated Gain](/posts/2026-07-06-looped-transformer-router-finally-starts-to-work/).

![Controlled fixed vs routed 2x4](/images/looped-transformer/controlled-language-router-comparison-plus-interleaved-20260701.png "Controlled fixed vs routed 2x4")

## The research question

The project is testing a small idea with a large implication:

```text
Can later block outputs feed back into early blocks, and can a learned router
decide when and where to route based on the input?
```

A normal Transformer spends depth in a straight line. A looped Transformer reuses a small number of blocks multiple times. The fixed version is simple: for example, two unique blocks run for four loops, giving an effective depth of eight block applications.

The routed version asks for something more interesting. Instead of always following the same fixed path, a router can choose:

- go to block 0
- go to block 1
- exit

The hope is that a routed model can preserve most of the gain from deeper computation while spending compute more selectively.

## Why this experiment used language-like data

Earlier looped-transformer experiments used pointer-chasing style tasks. Those are useful because they expose whether recurrence can solve multi-hop computation. But they are still classification-style probes.

The current experiment uses `language_composition`, a synthetic language-like next-token task. A prompt encodes a small composition problem, and the model predicts the answer token.

This is not natural-language corpus pretraining. It should not be sold that way. But it is closer to language-model training in the one way that matters for this stage: the model is trained through next-token prediction, and the benefit of extra computation has to show up through eval loss and accuracy.

That makes it a useful bridge between clean algorithmic probes and real language pretraining.

## The controlled grid

We compared two architectures:

| Architecture | Description |
| --- | --- |
| `fixed_2x4` | two unique blocks, four fixed loops |
| `router_2x4_step_path020` | two unique blocks, routed path, step signal, path loss weight `0.20` |

The grid controlled model width, training budget, and seed. The first grid used `d_model={32,48,64}`, budgets `{200,600,1200}`, and seeds `{0,1}`. The second interleaved grid added `d_model={40,56,80}`, budgets `{400,800,1600}`, and seed `0`.

| Axis | Values |
| --- | --- |
| `d_model` | `32`, `40`, `48`, `56`, `64`, `80` |
| steps | `200`, `400`, `600`, `800`, `1200`, `1600` |
| seeds | `0`, `1` for the original grid; `0` for the interleaved additions |
| task | `language_composition` |
| train max hops | `4` |
| eval max hops | `6` |
| num nodes | `32` |

That gives:

```text
36 original runs + 18 interleaved runs = 54 runs
```

The source CSV is:

```text
outputs/controlled_language_scaling_rows_seed0_seed1_plus_interleaved_20260701.csv
```

## What happened

The router looks best at medium training budgets:

| Budget | Mean router loss minus fixed loss | Router wins |
| --- | ---: | ---: |
| `200` steps | `+0.00104` | `3/6` |
| `400` steps | `-0.00595` | `3/3` |
| `600` steps | `-0.01149` | `6/6` |
| `800` steps | `-0.01906` | `3/3` |
| `1200` steps | `+0.00633` | `1/6` |
| `1600` steps | `+0.01311` | `1/3` |

Negative delta is better for the router. The `400`, `600`, and `800` step results are real enough to keep studying the router. But the `1200` and `1600` step results prevent us from promoting it as a stable architecture win.

By width:

| Width | Mean router loss minus fixed loss | Router wins |
| --- | ---: | ---: |
| `d32` | `-0.00347` | `5/6` |
| `d40` | `-0.00029` | `2/3` |
| `d48` | `-0.00230` | `2/6` |
| `d56` | `+0.00044` | `2/3` |
| `d64` | `+0.00165` | `3/6` |
| `d80` | `-0.01206` | `3/3` |

The old largest replicated point remains a stress test because it has two seeds:

| Seed | Fixed loss | Router loss | Router delta |
| --- | ---: | ---: | ---: |
| `0` | `3.272328` | `3.287227` | `+0.014899` |
| `1` | `3.225777` | `3.236505` | `+0.010727` |

The new largest point is `d_model=80`, `1600` steps, seed `0`:

| Seed | Fixed loss | Router loss | Router delta |
| --- | ---: | ---: | ---: |
| `0` | `3.141292` | `3.140688` | `-0.000603` |

That is encouraging, but it is not enough to declare a win. The router is basically tied at the largest new point, while the two smaller `1600`-step runs lose by `+0.02149` and `+0.01844`. So the current router is not simply undertrained. It can win in the middle, and it may recover at larger width, but its long-horizon behavior is still unstable.

## The router collapsed toward exit

The router statistics explain why the long-run result is weak.

The final exit mass increases as training gets longer:

| Width and budget | Final exit mass | Route entropy |
| --- | ---: | ---: |
| `d32 / 200 / seed 1` | `0.6578` | `0.3856` |
| `d32 / 1200 / seed 1` | `0.9616` | `0.0493` |
| `d40 / 400 / seed 0` | `0.7743` | `0.2038` |
| `d40 / 1600 / seed 0` | `0.9879` | `0.0255` |
| `d56 / 1600 / seed 0` | `0.9922` | `0.0200` |
| `d64 / 1200 / seed 1` | `0.9944` | `0.0236` |
| `d80 / 400 / seed 0` | `0.9131` | `0.0712` |
| `d80 / 800 / seed 0` | `0.9824` | `0.0206` |
| `d80 / 1600 / seed 0` | `0.9914` | `0.0127` |

This is the key failure mode. A router that always routes to the same endpoint is not really using adaptive computation. It is learning a low-entropy shortcut.

The current sequence-level router sees:

```text
mean hidden state + last-token hidden state + route-step signal
```

That input is enough to make the router trainable. It is not enough to make the router reliably assign different computation to different parts of the sequence.

## The fitted loss relationship

The pairwise comparison tells us which model wins at each grid point. But that is not enough. A routed model has slightly more parameters than the fixed model at the same width, and the training budgets now differ by a factor of eight. To make the conclusion meaningful, we need to ask a more controlled question:

```text
After controlling for model complexity and training data, does the router shift
the eval-loss curve downward?
```

For this grid I fit eval loss as a function of:

- `N`: trainable parameter count, used as the model-complexity proxy
- `D`: training tokens
- `R`: router indicator, where `R=1` for routed and `R=0` for fixed

The fitted relationship is:

```text
L(N, D, R) =
  2.0000
  + 0.16765 * (N / 1e5)^(-0.35628)
  + 1.21467 * (D / 1e7)^(-0.08564)
  - 0.00083 * R
```

The fit quality is reasonable for a 54-point exploratory grid:

| Metric | Value |
| --- | ---: |
| observations | `54` |
| RMSE | `0.0477` loss |
| MAE | `0.0381` loss |
| R2 | `0.769` |

This should not be treated as a final scaling law. The range is small, and the
floor term is constrained by the limited data. But it is strong enough to
separate three effects: more parameters, more training tokens, and the router
indicator.

![Controlled scaling-law fit](/images/looped-transformer/controlled-language-scaling-law-fit-plus-interleaved-20260701.png "Controlled scaling-law fit")

## What the fit says

First, eval loss improves with both model complexity and training data.

Using the fitted curve, increasing model size from the smallest fixed model to the
largest routed-width model scale at the largest token budget predicts about `0.1188` loss
improvement:

```text
N: 25.5K -> 126.6K params, D = 34.0M tokens
predicted loss improvement: 0.1188
```

Increasing data from the smallest to largest token budget at the largest width
predicts about `0.2132` loss improvement:

```text
D: 4.25M -> 34.0M tokens, N = 126.6K params
predicted loss improvement: 0.2132
```

So, in this small regime, the data axis is at least as important as the
model-complexity axis. That matters for the research plan: if we only compare
architectures at one short training horizon, we can easily mistake an
optimization-speed effect for a real architecture effect.

Second, after controlling for `N` and `D`, the router coefficient is slightly
negative but still basically zero relative to run noise:

```text
router offset = -0.00083 loss
```

That is about `1.7%` of the fit RMSE. In plain language, the enlarged grid gives
the router a tiny fitted advantage, but not a meaningful frontier shift yet. The
routed points mostly sit on the same loss surface as the fixed points.

That changes the conclusion. The right statement is not:

```text
the router sometimes wins
```

The right statement is:

```text
the current router has not yet shown a stable, material improvement to the
loss-vs-data-vs-complexity frontier.
```

This is exactly the kind of negative result worth keeping. It tells us the next
router needs a better mechanism, not just a wider grid.

## Why this is still encouraging

This result is not a failure of the looped-transformer idea. It is a failure of the first coarse router.

The fixed `2x4` model remains a strong anchor. It is small, simple, and improves with both width and data. That means the looped recurrent structure is still a real object of study.

The router also did not fail randomly. It wins most medium-budget paired points, then weakens when training is longer. The one exception is the new `d80 / 1600` point, where it is essentially tied with a tiny win. That pattern gives us a specific mechanism to fix:

```text
the router finds a shortcut before it learns a durable computation policy
```

A good next candidate should make route collapse harder and make useful compute allocation easier.

## The next router candidate

The next version should stay simple. We do not need a large Mixture-of-Recursion clone yet. We need the smallest router that addresses the observed failure.

The next candidate should add three things:

1. A target-compute pressure, so the router is not rewarded for collapsing too quickly.
2. An entropy floor or anti-collapse term, so all routes do not become identical early in training.
3. Token-conditioned routing with global context, so the router can use local token state while still seeing sequence-level difficulty.

The minimal design is:

```text
token hidden state
  + sequence summary
  + route-step embedding
  -> per-token continue / route weights
```

This keeps the useful part of the current router, the sequence-level context, but moves the routing decision closer to the token representation.

## Decision policy from here

This experiment also changes how we should make decisions.

Do not promote a router from a single seed or one training horizon. The minimum gate should be:

1. `600` steps, at least 2 seeds, controlled fixed-vs-router comparison.
2. `1200` and `1600` steps when the medium-budget result is close or horizon-sensitive.
3. Router stats must show nontrivial route entropy and noncollapsed exit mass.
4. The largest controlled point must not reverse the conclusion, and any apparent win should be replicated with a second seed.

By that policy, the current router is not promoted. It remains a useful baseline and a diagnostic tool.

## Bottom line

The best current statement is:

```text
Fixed 2x4 is the current controlled anchor. The sequence-level router is
competitive and maybe slightly helpful, but still unstable. The next real
candidate should be anti-collapse and token-conditioned.
```

That is not as exciting as declaring a win. It is more useful. The controlled language grid found the exact thing we need to improve before spending more compute: the router must learn a computation policy, not just an exit policy.

As a research sequence, this post is the failed mechanism. The later token-feedback result is the first candidate mechanism. Keeping both matters: without this controlled collapse result, the successful sparse-feedback design looks arbitrary; with it, the design constraint is obvious. The router has to be token-aware, sparse, and protected against damaging the broad language-model objective.
