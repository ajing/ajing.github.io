---
author: Jing Lu
pubDatetime: 2026-08-14T00:00:00-07:00
modDatetime: 2026-08-17T00:00:00-07:00
title: "Do Verifier Errors Grow Superlinearly with Horizon? A Six-Stage Experiment"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Reinforcement Learning
  - Evaluation
  - Post Training
description: "A preregistered 224-artifact study confirms a 66% hybrid-verifier improvement, does not support the universal superlinear headline, and finds that task structure can reverse the apparent horizon effect."
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

After six stages, including a two-seed confirmatory run, the result is more
specific than the original hypothesis:

- no semantic verifier passes the preregistered superlinearity test;
- a frozen structural-gate plus semantic-majority hybrid reduces controlled
  false-positive reward by **66.3%**, with a paired 95% interval of
  **55.6–76.1%**;
- the apparent horizon effect reverses across task families, so horizon alone
  is not a portable scaling variable;
- same-model errors are modestly correlated, but a different-model judge is
  not automatically safer;
- and the semantic confidence outputs fail their own measurement contract, so
  the calibration metric is not interpretable.

The narrow conclusion is not that long-horizon verifier error universally
grows superlinearly. It is that **verifier failure depends on the interaction
between horizon and evidence structure—and that a prospectively frozen
ensemble can still remove most false rewards**.

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

## V4: repair the denominator, expose the hybrid confound

V4 implemented the controlled-error design that V3 had motivated. Every task
produced two paired lanes:

- the natural actor artifact;
- a controlled artifact that was schema-valid, checksum-consistent, polished,
  and definitively wrong under replay.

This made the primary denominator independent of whether the actor happened
to fail in a particular cell. It also revealed a problem in the original
hybrid. The different-model judge falsely accepted `7/16` controlled errors,
while the hybrid falsely accepted `8/16`. On the seven naturally correct
artifacts, the hybrid rejected three.

The reason matters. The hybrid was not merely applying a deterministic gate to
the same semantic decision. It called the semantic model again, so its apparent
effect mixed the gate with fresh sampling variance. V4 was a valid negative
result: the denominator repair worked, but the intervention did not.

## V5: freeze a zero-additional-call majority hybrid

Before a new seed was run, V5 froze a different hybrid:

```text
accept = structural_integrity_passes
         AND at_least_two_of(self, same_model, different_model)_accept
```

This reuses the three semantic judgments already collected for comparison and
does not introduce a fourth stochastic judge call. On seed 97, the
different-model false-positive rate was `6/16`; the frozen hybrid reduced it to
`2/16`, a `66.7%` relative decrease. The paired pilot interval was wide
(`26.6–100%`), but the natural-artifact false-negative guardrail was `0/8`.

V5 was encouraging, not confirmatory. Each horizon still contained only four
controlled errors. Its purpose was to freeze the rule before the larger run.

## V6: the confirmatory controlled-error experiment

V6 used seeds 101 and 103 with no actor generation. It crossed:

```text
2 task families
× 4 horizons
× 2 difficulty tiers
× 7 artifacts per cell
× 2 seeds
= 224 controlled semantic errors
```

Every artifact was internally well-formed but wrong under authoritative
replay. All `672/672` semantic model calls succeeded. Each seed passed the
frozen local task-visible token gate at a `1.000×` max/min ratio, and each
family × horizon cell contained 28 wrong artifacts—above the preregistered
minimum of 15.

Because all V6 artifacts are wrong, verifier error, false-positive reward, and
reward-hacking rate are the same endpoint:

| Verifier                  |  h=20 |  h=50 | h=100 | h=200 | Overall |
| ------------------------- | ----: | ----: | ----: | ----: | ------: |
| Actor self-judge          | 12.5% | 10.7% | 17.9% | 25.0% |   16.5% |
| Fresh same-model judge    |  1.8% | 12.5% | 21.4% | 21.4% |   14.3% |
| Different-model judge     | 42.9% | 39.3% | 42.9% | 50.0% |   43.8% |
| Deterministic full replay |  0.0% |  0.0% |  0.0% |  0.0% |    0.0% |
| Frozen majority hybrid    | 14.3% |  8.9% | 16.1% | 19.6% |   14.7% |

![Verifier error by horizon in the V6 controlled-error experiment](/images/verifier-horizon/error-scaling-v6.svg "V6 verifier error by trajectory horizon")

### The hybrid result survives confirmation

The different-model judge falsely accepted `98/224` artifacts (`43.75%`). The
frozen hybrid accepted `33/224` (`14.73%`). The relative reduction is
`66.33%`, with a paired episode-bootstrap 95% interval of
`55.56–76.14%`.

The entire interval exceeds the preregistered 30% target. This is the strongest
positive result in the project.

There are two important cost qualifiers. First, V6 is controlled-error only,
so it cannot estimate false negatives; the last natural-lane guardrail remains
V5's small `0/8` result. Second, the majority hybrid requires three semantic
judgments in deployment. It adds no calls to this comparison panel, but it is
not a single-judge-cost verifier.

### The universal superlinear hypothesis does not survive

The preregistered test still requires both a power-law exponent whose 95%
interval is above one and quadratic curvature whose interval is above zero:

| Verifier               | Power exponent (95% CI) |      Curvature (95% CI) | Passes both? |
| ---------------------- | ----------------------: | ----------------------: | :----------: |
| Actor self-judge       |    0.310 [0.018, 0.675] |   0.046 [-0.501, 0.555] |      No      |
| Fresh same-model judge |    0.933 [0.488, 1.514] | -0.549 [-1.058, -0.063] |      No      |
| Different-model judge  |   0.066 [-0.085, 0.230] |   0.184 [-0.516, 0.898] |      No      |
| Majority hybrid        |   0.173 [-0.128, 0.537] |   0.086 [-0.465, 0.580] |      No      |

The same-model curve rises from `1.8%` at 20 steps to `21.4%` at 100 steps and
then saturates. A quadratic model fits better than a linear one, but its
curvature is significantly negative. That is a concave saturation pattern,
not the preregistered convex acceleration.

## The key result is a task-family interaction

The pooled curve hides two opposing regimes:

| Task family             | Verifier   |  h=20 |  h=50 | h=100 | h=200 |
| ----------------------- | ---------- | ----: | ----: | ----: | ----: |
| Register reconstruction | Self       |  0.0% |  3.6% | 28.6% | 50.0% |
| Register reconstruction | Same model |  0.0% | 10.7% | 35.7% | 42.9% |
| Register reconstruction | Hybrid     |  0.0% |  3.6% | 25.0% | 39.3% |
| File-manifest migration | Self       | 25.0% | 17.9% |  7.1% |  0.0% |
| File-manifest migration | Same model |  3.6% | 14.3% |  7.1% |  0.0% |
| File-manifest migration | Hybrid     | 28.6% | 14.3% |  7.1% |  0.0% |

For register reconstruction, the fitted power exponents for self, same-model,
and hybrid verification are all above `1.44`, with intervals entirely above
one. But their curvature intervals do not clear zero. The pattern is rapid
growth followed by saturation, not confirmed convex growth across all four
horizons.

For file-manifest migration, those same error rates decline. This sign reversal
means that “number of steps” is not a sufficient causal description. Evidence
topology, state-transition structure, where the lure appears, and how ambiguity
interacts with the output representation can dominate the nominal horizon.

This is a more useful research direction than adding more samples to the
pooled curve. The next experiment should factorially manipulate these structural
variables.

## Common-mode error is present, but not uniform

On the controlled lane, binary error correlations were:

| Judge pair                    | Error correlation | Joint error rate |
| ----------------------------- | ----------------: | ---------------: |
| Self vs fresh same model      |             0.334 |            6.70% |
| Self vs different model       |             0.117 |            9.38% |
| Same model vs different model |            -0.026 |            5.80% |

The self/same-model pair shows modest shared failure. The different-model judge
is nearly uncorrelated with the same-model judge in the pooled data, yet it has
the highest standalone false-positive rate. Independence is useful for an
ensemble; it is not the same thing as accuracy.

Correlations involving the hybrid are not independent evidence because the
hybrid is constructed from the three component decisions.

## The calibration metric failed its measurement contract

The semantic prompt defined confidence as the probability that the artifact is
fully correct. In V6, however, every semantic judgment returned a value between
`0.99` and `1.00`—including every rejection:

| Verifier               | Rejections | Rejections with confidence > 0.5 |
| ---------------------- | ---------: | -------------------------------: |
| Actor self-judge       |        187 |                              187 |
| Fresh same-model judge |        192 |                              192 |
| Different-model judge  |        126 |                              126 |

A model cannot coherently say “this artifact is wrong” while assigning it a
99% probability of being fully correct. The resulting ECE and Brier values
diagnose a broken output contract, not calibrated uncertainty. I therefore do
not treat calibration as a valid V6 result. The binary false-positive and
correlation results do not depend on that field and remain usable.

A future schema should separate `p_correct` from `decision_confidence`, reject
logically inconsistent responses, and calibrate only `p_correct`.

## What the six stages establish

The project now supports three claims with different strengths:

1. **Confirmed engineering result.** A prospectively frozen structural plus
   semantic-majority hybrid reduced controlled false-positive reward by 66.3%,
   with the paired interval wholly above the 30% target.
2. **Unsupported headline.** No verifier passed the preregistered universal
   superlinearity test. The data do not justify claiming that verifier error
   universally grows superlinearly with trajectory length.
3. **New mechanism hypothesis.** Horizon interacts with task and evidence
   structure strongly enough to reverse the observed slope across two task
   families.

The scope remains deliberately narrow. These are synthetic, exactly replayable
tasks; the controlled artifacts are benchmark-generated rather than natural
actor errors; Terra and Luna are different model IDs inside the same provider
stack; V6 has no correct-artifact lane; and the deterministic replay verifier
is an oracle-backed upper-bound control unavailable for most open-ended work.

The next confirmatory study should therefore add:

- a genuinely independent model provider;
- a balanced correct-artifact lane for false-negative estimation;
- separate factorial interventions for stale state, conflicting evidence,
  wrong-path artifacts, semantic examples, and rubric ambiguity;
- a corrected probability schema;
- and a human audit of disagreements near the hybrid threshold.

## Final takeaway

The answer to the title question is now sharper: **not as a universal law**.
Verifier error can rise rapidly with horizon, but the direction and shape depend
on what the extra steps do to the evidence available to the judge.

At the same time, the intervention result is real within this benchmark. A
frozen majority hybrid removed roughly two-thirds of false rewards without
using the formal replay oracle as its final semantic decision.

In long-horizon RL, the verifier is part of the learning environment. Treating
its denominator, token budget, confidence semantics, and task structure as
first-class experimental variables is what separates a convincing curve from a
reliable result.
