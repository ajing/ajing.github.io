---
author: Jing Lu
pubDatetime: 2026-07-17T19:30:00Z
modDatetime: 2026-07-25T05:30:00Z
title: "Reproducing CompactRL: What Worked, What Failed, and Why We Did Not Scale"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Reinforcement Learning
  - Post Training
description: "An auditable CompactRL reproduction spanning the public algorithm, a 96-step long-horizon simulation, integration with slime, real Qwen actor-critic training, value-function fixes, 17 experimental phases, and the evidence that stopped us from scaling."
---

Long-horizon agents eventually face a basic systems problem: their interaction
history can exceed the model's context window before the task is complete.
Summarizing old context keeps the agent running, but an inference-time summary
is not automatically a good reinforcement-learning action. If it drops the one
fact needed 50 turns later, the policy may never learn that the summary caused
the eventual failure.

[CompactionRL](https://arxiv.org/abs/2607.05378) makes a sharper proposal:

> Treat context compaction as an action sampled by the trainable policy, and
> give the summary tokens credit from the final task reward.

I built an open reproduction to test that idea from three directions:

1. implement and independently test the public equations;
2. test whether learned summaries solve a controlled 96-step task with three
   context resets;
3. integrate the method with public
   [THUDM/slime](https://github.com/THUDM/slime) and run real Qwen actor–critic
   training on Modal.

The complete code, configs, reports, and decision artifacts are public in
[`ajing/compactionrl-repro`](https://github.com/ajing/compactionrl-repro).

The short conclusion is:

- the algorithmic path is correct;
- the controlled long-horizon experiment works;
- the real LLM training path works;
- stable small-model capability improvement is **not** closed;
- the final experiments did not justify further scale-up.

This post explains both the positive result and the evidence that stopped us
from turning it into a larger claim.

## What “the GLM-5.2 technical report” actually refers to

The public evidence is split across three sources:

1. The [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2) describes
   the model and training pipeline.
2. The [GLM-5 technical report](https://arxiv.org/abs/2602.15763) describes
   asynchronous agent RL and the slime infrastructure in more detail.
3. The [CompactionRL paper](https://arxiv.org/abs/2607.05378) specifies the
   compaction rollout, loss, and cross-trajectory credit equations reproduced
   here.

There is no single public “GLM-5.2 technical report” containing a directly
executable CompactRL recipe. This project implements the intersection of the
three public sources and explicitly separates it from undisclosed prompts,
dataset ordering, reward code, and large-model training settings.

It does **not** claim to reproduce the paper's 30B or 106B benchmark scores.

## The algorithm in one pass

Let the active history be

$$
h_t = (\text{system}, \text{user}, (a_1,o_1), \ldots, (a_t,o_t)),
$$

where each assistant action $a_i$ and environment observation $o_i$ form one
atomic step. If the model's context capacity is $C$ and the required remaining
budget is $T_{\text{comp}}$, compaction triggers when

$$
C - |h_t| < T_{\text{comp}}.
$$

The same policy then samples a summary. Execution resumes from

$$
\text{system} + \text{resume}(\text{summary})
+ \text{newest } k \text{ atomic steps}.
$$

One logical trajectory becomes several trainable segments:

```text
execution_0
summary_0
execution_1
summary_1
execution_2
```

Four details are especially easy to get wrong:

- **The summary is an action.** It receives actor loss when the policy samples
  it.
- **The copied summary is context.** Its later appearance in the resume prompt
  must be masked so it does not receive loss twice.
- **Tool calls and observations remain atomic.** Splitting a pair corrupts
  environment state and token provenance.
- **Credit does not reset at segment boundaries.** Earlier summary tokens must
  remain connected to the eventual task reward.

### Cross-trajectory GAE

The paper computes local GAE inside each segment, then discounts earlier
segments by the number of optimized tokens that occur after them. For segment
$s$:

$$
\widehat A_{s,i}
=
(\gamma \lambda_s)^{N_{\text{after}}(s)}
\widehat A^{\text{local}}_{s,i}.
$$

The length-adaptive trace parameter is

$$
\lambda_s = 1 - \frac{1}{1.5L_s},
$$

where $L_s$ is the response length.

### Token-normalized PPO

Variable segment counts and lengths create another trap. Averaging one loss per
segment lets a short summary receive the same weight as a long execution
sequence. The implementation instead reduces once over every enabled token in
the global batch:

$$
\mathcal L
=
\frac{
\sum_{s,i} m_{s,i}\,\ell^{\text{PPO}}_{s,i}
}{
\sum_{s,i} m_{s,i}
}.
$$

These are not cosmetic details. The paper's 106B compacted ablation scores
`66.8/24.5` with the full method, `60.0/21.3` without token-level loss
normalization, and `63.0/22.5` without cross-trajectory GAE correction.

The full derivation and disclosure gaps are in the repository's
[technical analysis](https://github.com/ajing/compactionrl-repro/blob/main/docs/TECHNICAL_ANALYSIS.md).

## What the paper reports

The public results motivate this reproduction; they are not directly
comparable to our 0.5B synthetic-task experiments.

| Model and benchmark       | Base | PPO without compaction | CompactionRL |
| ------------------------- | ---: | ---------------------: | -----------: |
| 30B / SWE-bench Verified  | 50.5 |                   48.0 |         56.0 |
| 30B / Terminal-Bench 2.0  | 13.4 |                   12.4 |         20.2 |
| 106B / SWE-bench Verified | 59.8 |                   62.5 |         66.8 |
| 106B / Terminal-Bench 2.0 | 21.4 |                   23.6 |         24.5 |

The narrow supported conclusion is that compaction-aware RL beats the tested
baselines when evaluation also uses compaction under the same peak-context
constraint. It does not establish universal superiority.

An exact reproduction is blocked by missing details, including literal summary
and resume prompts, exact training rows and order, maximum summary length,
verifier and timeout behavior, optimizer settings, critic initialization, and
the internal GLM-5.2 configuration.

## The central question: what should the value function learn?

This was the easiest place to obtain a training run that appeared healthy while
delivering a corrupted actor signal.

CompactRL does not require a critic whose target is “summary quality.” The
default critic still predicts expected **task return**:

$$
V_\phi(s_t)\approx \mathbb E[R_{\text{task}}\mid s_t].
$$

Summary and execution tokens belong to the same terminal-reward chain. We
compute task returns and GAE first, then apply reward-distance correction to
summary advantages across segment boundaries. Compaction changes trajectory
structure and credit distance; it does not change the critic's semantic target.

The implementation maintains these invariants:

- execution and summary segments share the terminal task reward;
- the critic evaluates trainable summary states;
- copied context does not receive duplicate policy loss;
- critic warm-up may precede actor training;
- actor and critic checkpoints use separate paths;
- a missing scalar value head is **zero-initialized**, not randomized before
  PPO.

That last issue caused a concrete failure. On an all-zero-reward replay, a
random value head predicted approximately `0.9747`, produced critic gradient
norms as high as `785.3`, and even produced an actor gradient norm of `47.8`.
After zero initialization, the same audit yielded exactly zero values, returns,
advantages, and gradients.

### Why later summary-quality signals are actor-only

Phases 7–17 tested fidelity rewards, positive routing, token-local credit,
coverage-stop pressure, and causal-prefix redistribution. The resulting rule
is:

> The task critic learns only the environment-defined task return. Handcrafted
> summary-quality signals enter as optional actor-only auxiliary advantages
> after task GAE has been computed.

A repeated field, missing milestone, or excess summary token is not itself an
environment value. Mixing those heuristics into returns can produce a critic
that accurately fits the wrong target.

The corrected exact-paired Phase 17 audit validates this boundary. All 48 task
streams, outcomes, and token trajectories match Phase 16 exactly. Every
critic/value metric is bit-identical. Only actor loss and gradient change, by
the small amount predicted by the auxiliary intervention.

The current data flow is:

```text
environment reward
    → task return and GAE
    → critic target
    → cross-trajectory correction
    → optional actor-only summary credit
    → PPO actor loss
```

## A controlled 96-step long-horizon experiment

Before renting GPUs, I built a dependency-free memory chain:

- each episode lasts 96 steps;
- the context resets at steps 24, 48, and 72;
- only the newest two atomic steps survive outside the summary;
- the final answer depends on a fact observed at step 1;
- the joint arm trains execution and summary decisions;
- the control uses the same actor–critic but freezes the summary policy.

Across seeds `7`, `17`, and `29`:

| Method                 | Mean accuracy |
| ---------------------- | ------------: |
| Frozen-summary control |        51.81% |
| Joint CompactionRL     |        99.69% |
| Difference             |     +47.88 pp |

This is not a language-model benchmark. It answers a narrower causal question:
can terminal reward traverse multiple compaction boundaries and teach an early
summary action to preserve task-critical state? In this setting, yes.

The
[experiment](https://github.com/ajing/compactionrl-repro/blob/main/experiments/long_horizon_memory_chain.py)
and its
[readable trace](https://github.com/ajing/compactionrl-repro/blob/main/results/long_horizon_memory_chain.md)
are both public.

## Integrating with public slime

The audited slime revision is
[`fb42ae4`](https://github.com/THUDM/slime/tree/fb42ae456fac8166afb604f13b30d22bb3c75053).
It already provides PPO, a critic, per-token loss, a custom advantage hook,
`list[Sample]` fan-out, and agent adapters. At that revision, it does not
provide a directly enabled “CompactionRL estimator plus rollout.”

The reproduction therefore uses a thin integration:

- framework-neutral equations live in `src/compactionrl/`;
- rollout and framework adapters live in `integrations/slime/`;
- custom advantages enter through a public extension point;
- a minimal patch is pinned to the audited critic loop;
- each live run records immutable identity, token trajectories, masks,
  advantage/return audits, actor/critic gradients, and role-specific
  checkpoints.

The concrete contract and launch path are documented in the
[slime integration guide](https://github.com/ajing/compactionrl-repro/blob/main/docs/SLIME_INTEGRATION.md).

## Calibration before the numbered phases

The later experiments did not begin from an unverified GPU stack. Earlier
controls progressively isolated sampling, checkpoint, grammar, and
full-context learnability:

| Experiment                         | Result                                                                | Interpretation                                    |
| ---------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| First five-update scale pilot      | Full `7/40`; summary-masked `5/40`; both `0/23` on `early_fact`       | E2E path works; no retention evidence             |
| Corrected-value three-arm training | 60 updates and 50 critic warm-up steps, but roles shared a save path  | Training works; old checkpoint claim invalid      |
| Role-specific checkpoint eval      | Trained `12/60`; base `14/60`                                         | Two actor updates do not improve held-out ability |
| Second seed continuation           | Seed 17 `20/60`; seed 23 `12/60`; base `14/60`                        | Positive result does not replicate across seeds   |
| Critic LR `3e-6→1e-6`              | Mean value loss roughly halves; score `12→13/60`, `p=1`               | Stability is not capability                       |
| Reference-KL screen                | KL `0.01` scores `14/60`; KL `0.05` initially scores `21/60`          | KL requires paired replication                    |
| Deterministic sampling             | Two independent 4×A10 runs match 8 trajectories and 183 request seeds | Common-random-number comparison is viable         |
| Controlled KL replication          | No-KL `26/60`; KL `0.05` `19/60`                                      | Initial KL gain is overturned                     |
| Independent no-KL rerun            | Original `26/60`; rerun `25/60`; base `14/60`                         | Same-seed gain repeats                            |
| Deterministic optimizer gate       | Tokens, actor gradients, and four critic updates match exactly        | Short optimizer comparisons are auditable         |
| Second model seed                  | Seed 17 no-KL `29/60`; base `14/60`; seed 23 is also positive         | No-KL signal appears on two model seeds           |
| Full-context control               | `40/60`; early/latest each `20/20`; ordered-plan `0/20`               | Ordered-plan first has a grammar problem          |
| Schema hint and fixed example      | Ordered-plan remains `0/20` and `1/20`                                | Prompt patches do not fix the grammar             |
| Grammar SFT                        | Full-context `20/20`; compacted `8/20`                                | A genuine summary-transfer gap is isolated        |

The reversal of the initial KL `0.05` result is particularly important.
Without a fixed task stream and common sampling randomness, an attractive RL
difference may simply represent two different training trajectories.

## Complete Phase 1–17 result ledger

`PASS` below means that a mechanism or preregistered gate passed. It does not
mean that the paper's model capability was reproduced.

| Phase | Intervention                                  | Key result                                                                                                                 | Decision                              |
| ----: | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
|     1 | First public slime/Qwen pilot                 | Full `7/40`; summary-masked `5/40`; both `0/23` on `early_fact`                                                            | Path PASS; capability unproven        |
|     2 | Reward-bearing RL after grammar SFT           | Seeds 23/29 reach `16/20` and `15/20`; locked test `16/20`, `13/20` vs SFT `6/20`; seed 37 regresses to `2/20`             | Signal on 2/3 seeds; unstable         |
|     3 | Fresh preregistered data, batch 24            | SFT `22/60`; trained seeds `31/60`, `41/60`, `36/60`; one seed fails invention gate                                        | Do not scale                          |
|     4 | Actor reference KL `0.01`                     | KL exact is below no-KL in both pairs: `25 vs 28` and `39 vs 52` out of 60                                                 | KL is not the default                 |
|     5 | Positive-trajectory actor advantages          | With positive samples, `44/60` vs batch-gate `42/60`; without them, both remain `19/60`                                    | Works but depends on sparse successes |
|     6 | Scaled positive advantages                    | Both seeds underperform batch gate: `50 vs 55` and `48 vs 52` out of 60                                                    | Reject scale-up                       |
|     7 | Scalar fidelity reward `y·q`                  | Critic fits correctly, but every failed summary has target zero; no stable gain                                            | Reward is too sparse                  |
|     8 | Dense signed reward `R=y+q-1`                 | Candidates collapse to `0/60` and `2/60` with `2.9/5.6`-token summaries; controls score `53/60`, `55/60`                   | Negative advantages damage actor      |
|     9 | Zero nonpositive actor advantages only        | Rescues the Phase 8 settings from `5/60`, `0/60` to `51/60`, `55/60`                                                       | Establish actor/value routing         |
|    10 | Admit summary loss only when `R>0.999`        | Perfect-fidelity `54/60` vs any-success `57/60`; repetition falls `93.75%→37.89%`, but absolute gates fail                 | Useful constraint; not closed         |
|    11 | Summary-only k3 KL `0.01`                     | `54/60` vs no-KL `50/60`; invention `7.53% vs 26.88%`; length `107.58 vs 138.93`; repetition still `65.59%`                | Best tradeoff; stop scaling           |
|    12 | Increase KL to `0.03`                         | Exact `55/60`, but invention `9.68%`, repetition `82.8%`, and length `131.67` worsen                                       | No full retrain                       |
|    13 | Actor-only scalar fidelity offset             | Exact `57/60`, but invention `13.98%`, repetition `88.17%`, and length `134.72`                                            | Scalar credit is too coarse           |
|    14 | Local supported/repeated-token credit         | First attempt exposes token re-encoding mismatch; retry is blocked by nested-diagnostic logging; no checkpoint             | Two engineering faults found          |
|    15 | Repair logger; close token-credit update      | Actor optimizer PASS; critic tensors unchanged; actor-only eval `55/60` vs Phase 11 `54/60`, `p=1`, with worse quality     | Mechanism PASS; capability FAIL       |
|    16 | Lower coefficient plus coverage-stop pressure | Dual checkpoints PASS; eval `52/60`, below Phases 11/15; repetition and length do not improve                              | Do not scale                          |
|    17 | Marker-gated causal-prefix credit             | Corrected pair has bit-identical critic metrics; actor loss `-0.0384168→-0.0384282`; gradient norm changes only `-0.0588%` | Mechanism PASS; no eval or scale      |

Every number traces to a
[phase report](https://github.com/ajing/compactionrl-repro/tree/main/reports)
and a
[frozen decision artifact](https://github.com/ajing/compactionrl-repro/tree/main/configs).

## Failures that could have produced false conclusions

### A random value head

The missing critic head was initially random, so even zero reward produced
large advantages. Zero initialization and a zero-reward replay gate fixed it.

### A shared actor/critic checkpoint directory

Critic shards saved later replaced actor shards, invalidating an early
saved-checkpoint capability claim. Role-specific paths and checkpoint inventory
are now publication gates.

### Re-encoding text does not reproduce sampled tokens

Phase 14 initially retokenized decoded summary text to locate credit. A real
tokenizer does not guarantee segment-wise decode–encode identity. The repair
aligns against sampled-token prefix decoding and audits every offset and its
total mass.

### Logging made valid training look like a failed mechanism

A nested diagnostic could not be serialized by the slime logger. Rollout and
critic work had completed, but the run produced no checkpoint. Phase 15 closed
the mechanism only after repairing the logger.

### The first Phase 17 run was not an exact pair

The launcher omitted the Phase 17 profile and routed it to `synthetic`, while
Phase 16 used `synthetic_phase10`. The completed run was rejected as paired
evidence. After freezing and testing dataset routing, the corrected run matched
all 48 tasks, outcomes, and trajectories.

These failures show why “RL is unstable” is not a sufficient explanation.
Dataset, sampling, value initialization, checkpoint, and logging errors must be
eliminated before the remaining difference can be called optimization
variance.

## Why we did not scale

Several local results were strong:

- deterministic no-KL seed 23: `26/60` vs base `14/60`, paired `p=0.00754`;
- independent same-seed rerun: `25/60`;
- seed 17: `29/60` vs base `14/60`, paired `p=0.000729`;
- substantial ordered-plan gains on two Phase 2 seeds;
- simultaneous exact, invention, and length improvement in Phase 11.

The stronger negative evidence matters just as much:

- the initial KL `0.05` gain reverses under common-random-number control;
- the first ordered-plan full-context control is `0/20` because of output
  grammar, not memory;
- once grammar SFT reaches `20/20` in full context, compaction reduces it to
  `8/20`, isolating summary transfer as the real problem;
- the third Phase 2 seed fails;
- Phase 3 fails its absolute invention gate;
- most Phase 11–17 interventions improve one metric while damaging repetition,
  invention, or length;
- Phase 17 changes the real actor gradient by only about `0.059%`.

Simultaneously satisfying all criteria is difficult because the current small
model and reward design operate in a multi-objective conflict region. That is
an experimental result, not a reason to lower the criteria. Adding GPUs would
sample the same conflict more expensively.

## Cost discipline

Later experiments used a staged authorization ladder:

1. local analytic or counterfactual gate;
2. tokenizer and dataset preflight;
3. one-update mechanism probe;
4. evaluation only after mechanism and quality gates pass;
5. multi-seed or scaled runs only after evaluation passes.

The conservative GPU cost ceilings for Phases 15, 16, and 17 were approximately
`$0.75`, `$0.69`, and `$1.07`. Phase 17 includes both the rejected
wrong-dataset run and the corrected pair. We did not purchase a Phase 17
evaluation because the exact-paired gradient audit already showed that the
effect was too small.

A cheap run is not justified merely because it costs less than one dollar. It
must answer one question that can change the next decision.

## Reproduce the reproduction

The core algorithm and long-horizon simulation require no third-party package:

```bash
git clone https://github.com/ajing/compactionrl-repro.git
cd compactionrl-repro
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 examples/toy_joint_training.py
PYTHONPATH=src python3 experiments/long_horizon_memory_chain.py
```

Generate and inspect the deterministic synthetic data:

```bash
PYTHONPATH=src python3 scripts/generate_sim_data.py
PYTHONPATH=src python3 scripts/summarize_sim_data.py
```

Validate a pinned slime checkout:

```bash
python3 scripts/check_slime_compat.py --slime-root /path/to/slime
```

For a real Modal run, do not begin with the largest profile. Read the preflight,
evidence-download, and publication-gate sections of the
[integration guide](https://github.com/ajing/compactionrl-repro/blob/main/docs/SLIME_INTEGRATION.md),
then select a frozen
[configuration](https://github.com/ajing/compactionrl-repro/tree/main/configs).

## What should come next

The next step is not to scale Phase 17 unchanged. More promising changes are:

1. keep the task critic pure and train a separate, calibrated summary-quality
   model used only as a constraint or actor-side control variate;
2. represent summaries as structured memory slots so write, overwrite, and
   stop become explicit actions;
3. begin with tasks already proven solvable in full context, then increase
   compaction count through a horizon curriculum;
4. preregister a cross-seed Pareto gate over task success, milestone recall,
   invention, repetition, and summary length;
5. move to a larger model only after the small model exhibits a stable and
   sufficiently large gradient effect.

## Bottom line

CompactRL's core idea is credible and reproducible: make the summary a policy
action and propagate terminal reward through context resets to summary tokens.
In the controlled 96-step experiment, it nearly perfectly solves a problem
that the frozen-summary control cannot solve. On public slime, the reproduction
also demonstrates an auditable path through actor, critic, GAE, masks,
checkpoints, and deterministic trajectories.

“The algorithm can work” and “this small-model recipe works robustly” are
different claims. The evidence supports the first. The second likely requires
a different summary representation or credit model, not a larger copy of
Phase 17.

The most useful artifact is not only a successful experiment. It is the
combination of successful and failed evidence that prevents the next
researcher from paying to rediscover the same mistakes.
