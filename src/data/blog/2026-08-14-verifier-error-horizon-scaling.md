---
author: Jing Lu
pubDatetime: 2026-08-14T00:00:00-07:00
title: "Do Verifier Errors Grow Superlinearly with Horizon? A Three-Stage Experiment"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Reinforcement Learning
  - Evaluation
  - Post Training
description: "A controlled long-horizon experiment found a clear horizon effect but no preregistered evidence of superlinear verifier error—and exposed why token measurement and false-negative costs matter before scaling RL evaluations."
---

Long-horizon reinforcement learning depends on a deceptively strong assumption:
after an actor completes a long task, a verifier can still tell whether the
result is correct.

That assumption becomes fragile when the trajectory contains conflicting
documents, stale state, convincing artifacts on the wrong path, semantically
wrong outputs with valid structure, or an ambiguous rubric. Even if the
verifier never sees the actor's narration, the actor and verifier may share the
same model family, data sources, provider, or interpretation of the task. Their
errors can therefore become correlated.

I built a controlled benchmark around one question:

> Does verifier error grow superlinearly as the task horizon increases from 20
> to 50, 100, and 200 steps?

The result is useful precisely because it is not a clean positive result:

- verifier error shows a strong horizon effect;
- no semantic verifier passes the preregistered superlinearity test;
- an integrity-gated hybrid eliminates observed false positives in the pilots;
- the same hybrid also rejects substantially more correct outputs;
- and the second experimental design failed because its token manipulation
  check measured more than the experiment intended.

The narrow conclusion is not that long-horizon verifier error is superlinear.
It is that **measuring the scaling law is harder than making the curve look
convincing**.

## Experimental setup

The benchmark contains two independently replayable task families:

1. **Register reconstruction.** Starting from six registers, the actor applies
   a journal of modular `add`, `multiply`, and `xor` operations.
2. **File-manifest migration.** The actor applies `create`, `update`, `move`,
   and `delete` events and reconstructs the final path-to-content manifest.

Both tasks have a machine oracle. The authoritative journal can be replayed to
recover the exact final state, so semantic correctness is not determined by a
second language model.

Every stress episode can contain five lures:

| Lure                      | Failure pressure                                                          |
| ------------------------- | ------------------------------------------------------------------------- |
| Conflicting documents     | A mirrored journal changes a small fraction of operands                   |
| Stale state               | A valid midpoint snapshot is labeled as the latest state                  |
| Wrong-path artifact       | A schema-valid candidate appears under an incorrect path                  |
| Semantically wrong output | A polished example has valid structure but the wrong state                |
| Ambiguous rubric          | The instructions suggest that a merely plausible answer may be acceptable |

I compared five verification strategies:

1. actor self-judge;
2. a fresh instance of the same model;
3. a different-model judge;
4. a deterministic verifier with access to the replay oracle;
5. a deterministic structural and integrity gate followed by an independent
   semantic judge.

The actor's raw narration was retained for audit but never included in any
judge prompt. Each semantic judgment ran in an isolated fresh session.

The main metrics were false-positive reward, false-negative reward,
calibration error, reward-hacking rate, verifier error as a function of
horizon, and actor–judge error correlation.

## V1: a horizon effect, but not a superlinear result

The first aggregate contains two seeds, 112 episodes, and 560 paired
judgments. Each task-family × horizon cell contains 14 episodes.

The error rates clearly worsen at the longest horizon:

| Horizon | Actor self-judge | Same-model judge | Different-model judge | Integrity hybrid |
| ------: | ---------------: | ---------------: | --------------------: | ---------------: |
|      20 |            10.7% |             7.1% |                 21.4% |            25.0% |
|      50 |             3.6% |            14.3% |                 21.4% |            21.4% |
|     100 |            10.7% |            14.3% |                 21.4% |            17.9% |
|     200 |            32.1% |            39.3% |                 39.3% |             3.6% |

The same-model verifier rises from `7.1%` error at 20 steps to `39.3%` at 200
steps. Actor self-judgment rises from `10.7%` to `32.1%`.

That is a substantial horizon effect. It is not, by itself, evidence of
superlinear growth.

### The preregistered test

I did not define “superlinear” as a line that appears to bend upward. For each
verifier, the analysis fits linear, quadratic, and power-law models. A semantic
verifier must satisfy both of the following after episode-level bootstrap:

1. the 95% confidence-interval lower bound for the power-law exponent is
   greater than `1`;
2. the 95% confidence-interval lower bound for quadratic curvature is greater
   than `0`.

No semantic verifier passes both tests.

For example, the same-model judge has a fitted power exponent of about `0.60`,
with a 95% interval spanning approximately `0.20–1.32`. Its quadratic
curvature estimate is positive, but the interval spans approximately
`−0.40–0.97`. The uncertainty still includes linear and sublinear behavior.

The publication gate also required at least 50 episodes per task-family ×
horizon cell. The experiment has 14. The honest result is therefore:

> V1 supports increasing difficulty and increasing verifier error at long
> horizons, but does not support the preregistered superlinear claim.

## The hybrid result looked excellent—until we included its full cost

The integrity hybrid first applies a cheap deterministic gate. It checks the
artifact schema and verifies that the reported checksum matches the candidate
state. It does not replay the authoritative journal; a self-consistent but
semantically wrong artifact still reaches the semantic judge.

In V1, the different-model judge had a `31.25%` false-positive rate on wrong
artifacts. The hybrid had `0%`, a 100% relative reduction.

This passed the original engineering target of at least a 30% reduction, but
the result has an important scope condition: the benchmark provides integrity
metadata that can be calculated independently. Open-ended tasks often do not.

More importantly, false-positive reduction is only half of the operating
point. A verifier can achieve zero false positives by rejecting everything.
Later pilots therefore made the false-negative cost explicit.

## V2: the experiment stopped because the token gate failed

V2 changed the primary endpoint to

```text
P(verifier accepts | actor artifact is wrong)
```

and placed all five lures into each stress episode. Its Stage A crossed two
task families, four horizons, and two difficulty tiers for 16 episodes.

Before scaling to Stage B, both actor and different-model judge prompts had to
remain within a preregistered `1.10×` max/min token ratio across cells.

Two pilots failed:

| Pilot           | Actor token ratio | Judge token ratio | Wrong artifacts | Different-model FPR | Hybrid FPR |
| --------------- | ----------------: | ----------------: | --------------: | ------------------: | ---------: |
| Initial         |            1.117× |            1.280× |         10 / 16 |               80.0% |         0% |
| Tier-calibrated |            1.149× |            1.212× |          9 / 16 |               55.6% |         0% |

The second pilot added more character padding but produced a worse actor ratio.
Following the preregistration, I stopped instead of expanding the run.

The failure was informative. Equal character counts are not equal tokenizer
counts, and the Codex CLI's reported `input_tokens` include runtime context
beyond the task text. That aggregate is useful for operational accounting, but
it is not a clean task-only manipulation check.

The [OpenAI token-counting guide](https://developers.openai.com/api/docs/guides/token-counting)
similarly distinguishes counting plain text from counting a complete structured
request, whose message formatting, tools, and schemas add tokens. The
[Codex CLI reference](https://developers.openai.com/codex/cli/reference/)
documents JSONL usage output but does not expose a task-only decomposition of
the runtime input count.

V2 therefore diagnosed a measurement problem. It did not produce new evidence
for the behavioral scaling law.

## V3: exact task-visible token control

V3 froze a new measurement rule before any effective model calls:

- count the exact plaintext that the experiment submits to the actor or judge;
- use a pinned `o200k_base` tokenizer asset;
- pad each candidate to an identical integer token target;
- treat the CLI's full runtime token count as a diagnostic, not the primary
  manipulation check;
- require a local max/min ratio of at most `1.001×`;
- run a zero-call preflight over every candidate prompt before Stage A.

The preflight and live run both passed:

| Check                                   | Result |
| --------------------------------------- | -----: |
| Actor experiment-owned plaintext ratio  | 1.000× |
| Different-model judge plaintext ratio   | 1.000× |
| Actor CLI runtime ratio                 | 1.004× |
| Different-model judge CLI runtime ratio | 1.004× |

This repairs the experiment-owned prompt-length manipulation. It does not
claim that the entire hidden runtime request is exactly equal across calls.
System instructions, message structure, tool schemas, and other CLI-owned
context remain outside the local proxy.

That distinction is the main methodological output of V3: **name the quantity
that is controlled instead of quietly treating a convenient token counter as
the causal variable**.

## V3 behavioral result: fewer false positives, more false negatives

The 16-episode V3 Stage A produced seven wrong artifacts and nine correct
artifacts.

On the wrong artifacts:

| Verifier                      | False-positive reward |
| ----------------------------- | --------------------: |
| Actor self-judge              |         28.6% (2 / 7) |
| Same-model judge              |         42.9% (3 / 7) |
| Different-model judge         |         57.1% (4 / 7) |
| Deterministic replay verifier |            0% (0 / 7) |
| Integrity hybrid              |            0% (0 / 7) |

The hybrid again achieves a 100% false-positive reduction relative to the
different-model judge.

On the nine correct artifacts, however:

| Verifier                      | False-negative reward |
| ----------------------------- | --------------------: |
| Actor self-judge              |         11.1% (1 / 9) |
| Same-model judge              |         11.1% (1 / 9) |
| Different-model judge         |         33.3% (3 / 9) |
| Deterministic replay verifier |            0% (0 / 9) |
| Integrity hybrid              |         55.6% (5 / 9) |

The hybrid is safer against false rewards and substantially more conservative
on correct work. It meets the false-positive reduction target, but it is not a
cost-free improvement.

This changes how the intervention should be evaluated. The relevant object is
not a single FPR number; it is a Pareto frontier between rewarding wrong work
and rejecting correct work.

## Why V3 still cannot estimate the horizon curve

The primary endpoint conditions on the actor being wrong. The wrong-artifact
counts by horizon were:

```text
h20:   1 / 4
h50:   0 / 4
h100:  3 / 4
h200:  3 / 4
```

At `h50`, the conditional endpoint has no denominator. Fitting or drawing a
smooth horizon curve would invent information that the experiment did not
observe.

V3 used 74 remote calls—16 actor calls and 58 semantic judgments—within its
preregistered cap of 80. It did not automatically enter Stage B.

## What I would change before Stage B

Simply running more episodes from the same generator is not the best next
step. The next design should repair the wrong-artifact denominator first.

### 1. Generate controlled actor errors

Each family × horizon cell should contain a preregistered minimum number of
wrong artifacts. One option is to pair natural actor outputs with controlled
semantic mutations whose wrongness is verified by the machine oracle.

This would estimate verifier behavior conditional on the same error classes at
every horizon, instead of allowing actor competence to determine whether a
cell has a denominator.

### 2. Report two primary views

The next experiment should retain both:

- unconditional verifier error over all episodes;
- conditional false acceptance over wrong artifacts.

The first captures the operational system; the second isolates verifier
susceptibility once the actor fails.

### 3. Add provider-level independence

Using different model IDs inside the same provider is useful but does not
eliminate shared training data, system prompts, serving infrastructure, or task
interpretations. A confirmatory run should cross model family and provider.

### 4. Treat the hybrid as a thresholded system

Instead of asking whether one fixed hybrid wins, vary the structural and
semantic rejection thresholds and estimate the FPR/FNR frontier. Human audit
should target disagreements near that frontier.

### 5. Keep stopping rules prospective

Token gates, minimum wrong counts, confidence-interval tests, and expansion
decisions should remain frozen before the calls they govern. Otherwise a
long-horizon experiment can become a long sequence of researcher degrees of
freedom.

## Final takeaway

The most tempting headline would be that verifier error grows superlinearly
with horizon. The data do not support that headline yet.

What the experiment does support is more operationally useful:

1. long horizons can sharply increase actor and semantic-verifier error;
2. shared-model and shared-provider judges do not automatically remove
   common-mode failure;
3. integrity constraints can eliminate observed false rewards in a controlled
   benchmark;
4. that gain may come with a large false-negative cost;
5. task-visible tokens and full runtime tokens are different measurements;
6. a conditional verifier curve is not estimable when the actor produces no
   errors in a cell.

In long-horizon RL, the verifier is part of the learning environment. If its
measurement, denominator, and operating point are not controlled, scaling the
number of trajectories may only make the wrong conclusion more precise.
