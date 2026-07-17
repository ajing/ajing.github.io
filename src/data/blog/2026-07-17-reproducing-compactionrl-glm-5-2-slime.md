---
author: Jing Lu
pubDatetime: 2026-07-17T19:30:00Z
title: "Reproducing CompactionRL: From the GLM-5.2 Algorithm to a Live slime E2E"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Reinforcement Learning
  - Post Training
description: "A source-grounded reproduction of CompactionRL: the algorithm, missing recipe details, a 96-step multi-seed causal experiment, and a live Qwen actor-critic update through THUDM/slime on four A10 GPUs."
---

Long-horizon agents have a basic systems problem: their interaction history can grow beyond the model's context window before the task is finished.

The GLM-5.2 materials motivated this investigation, but the public algorithmic specification is the separate CompactionRL paper. I treated the [GLM-5 technical report](https://arxiv.org/abs/2602.15763) and [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2) as model context, not as a complete training recipe.

The usual response is to summarize old context. But an inference-time summarizer is not automatically a good reinforcement-learning action. If a summary silently drops the one fact needed 50 turns later, the policy may never learn that the summary caused the eventual failure.

[CompactionRL](https://arxiv.org/abs/2607.05378) makes a sharper proposal:

> Treat context compaction itself as an action sampled by the trainable policy, and give those summary tokens credit from the final task reward.

That sounds simple. The hard part is preserving the RL semantics after one logical episode is split into several execution and summary sequences.

I built an open reproduction to answer two separate questions:

1. Does jointly training the summary channel solve a controlled long-horizon task where a frozen summary cannot?
2. Can the algorithm actually run through the public [THUDM/slime](https://github.com/THUDM/slime) stack with a real actor, critic, rollout engines, gradients, and checkpoint?

The answer to both is yes, within a deliberately narrow claim boundary. The code and evidence are public in [`ajing/compactionrl-repro`](https://github.com/ajing/compactionrl-repro). The immutable result bundle is summarized in the [final E2E report](https://github.com/ajing/compactionrl-repro/blob/main/results/final_e2e_report.md).

This is not a reproduction of the paper's GLM-scale benchmark gains. It is an algorithmic reproduction, a controlled causal experiment, and a live public-stack integration test.

---

## 1. The Algorithm in One Pass

Let the active history be:

$$
h_t = (\text{system}, \text{user}, (a_1,o_1), \ldots, (a_t,o_t)).
$$

Here, each assistant action $a_i$ and environment observation $o_i$ form one atomic step. Let the model's context capacity be $C$, and let $T_{\text{comp}}$ be the minimum remaining budget before compaction.

Compaction triggers when:

$$
C - |h_t| < T_{\text{comp}}.
$$

At that point, the same trainable policy samples a summary $S_t$ from a fixed summary instruction. Execution then resumes from:

$$
\text{system} + \text{resume}(S_t) + \text{last } k \text{ atomic steps}.
$$

The paper's default recent tail is $k=2$. If the reconstructed prompt is still too large, the implementation reduces $k$ until it fits. The original action/observation boundary must remain intact; cutting an observation away from the action that produced it changes the environment history.

### One episode becomes several trainable segments

A logical trajectory may now look like:

```text
execution_0
summary_0
execution_1
summary_1
execution_2
```

Every segment shares one logical rollout ID and the same terminal verifier reward. Only tokens sampled by the policy receive loss:

- assistant execution tokens: mask 1;
- summary-generation tokens: mask 1;
- prompts, environment observations, copied summaries, and copied recent-tail tokens: mask 0.

This token provenance rule is critical. A summary should receive policy loss when it is generated, exactly once. Its copied appearance inside a later resume prompt must not receive loss again.

### Cross-trajectory GAE

The paper computes local GAE inside each segment, then discounts earlier segments by the number of optimized tokens that occur after them.

For segment $s$, let $N_{\text{after}}(s)$ be the number of enabled policy tokens in later segments. The corrected advantage is:

$$
\widehat A_{s,i}
=
(\gamma \lambda_s)^{N_{\text{after}}(s)}
\widehat A^{\text{local}}_{s,i}.
$$

The reported length-adaptive trace parameter is:

$$
\lambda_s = 1 - \frac{1}{1.5L_s},
$$

where $L_s$ is the segment response length. The reproduction uses the number of optimized response tokens and exposes this choice as an ablation because the public description does not fully settle every boundary convention.

### Token-normalized PPO

Variable numbers and lengths of segments create another trap. Averaging one loss per segment lets short summaries receive the same weight as long execution sequences.

The reproduction instead sums over every enabled token in the global batch and divides once by the total enabled-token count:

$$
\mathcal L
=
\frac{
\sum_{s,i} m_{s,i}\,\ell^{\text{PPO}}_{s,i}
}{
\sum_{s,i} m_{s,i}
}.
$$

This is not a minor reducer choice. In the paper's 106B ablation, removing token-level loss hurts more than removing cross-trajectory GAE on the reported compacted evaluations.

---

## 2. What the Paper Shows—and What It Does Not Release

The paper reports the following same-scaffold compacted-evaluation results:

| Model                   | Base | No-compaction PPO | CompactionRL |
| ----------------------- | ---: | ----------------: | -----------: |
| 30B SWE-bench Verified  | 50.5 |              48.0 |     **56.0** |
| 30B Terminal-Bench 2.0  | 13.4 |              12.4 |     **20.2** |
| 106B SWE-bench Verified | 59.8 |              62.5 |     **66.8** |
| 106B Terminal-Bench 2.0 | 21.4 |              23.6 |     **24.5** |

For the 106B model, the reported ablations are:

| Variant                      | SWE-Verified | Terminal-Bench 2.0 |
| ---------------------------- | -----------: | -----------------: |
| Full CompactionRL            |     **66.8** |           **24.5** |
| Without token-level loss     |         60.0 |               21.3 |
| Without cross-trajectory GAE |         63.0 |               22.5 |

The narrow supported conclusion is that compaction-aware RL beats the tested baselines when evaluation also uses compaction under the same peak-context constraint. It does not establish universal superiority. Disabling compaction or changing the resume protocol at test time creates a distribution shift.

An exact reproduction is also blocked by missing details, including:

- literal summary and resume prompts;
- exact training rows, ordering, duration, and seeds;
- maximum summary length and summary sampling settings;
- verifier and timeout handling;
- several PPO, value, KL, optimizer, and warmup settings;
- the exact interpretation of length for adaptive $\lambda$;
- critic initialization and cross-boundary bootstrapping details;
- the internal GLM-5.2 CompactionRL configuration and checkpoints.

So the right goal was not to manufacture a fake "paper reproduction" number. It was to isolate the public algorithm, test its causal claim, and prove the implementation against a real open RL stack.

---

## 3. A Controlled Long-Horizon Experiment

The repository includes a deterministic sequential-records environment with 360 tasks split into 240 train, 60 development, and 60 test examples. Tasks contain relevant updates, distractors, and an exact terminal verifier.

The stronger experiment is a 96-step memory chain:

- a hidden fact appears at step 1;
- execution continues for 96 steps;
- compaction is forced at steps 24, 48, and 72;
- only the newest two atomic steps survive outside the summary;
- the final answer depends on the step-1 fact.

The recent tail therefore cannot solve the task. Information has to pass through the learned summary at every reset.

I compared:

1. **Joint CompactionRL:** both summary and execution decisions learn from the terminal reward.
2. **Frozen summary:** the execution policy learns, but the summary policy remains fixed.

Across seeds 7, 17, and 29:

| Arm                |      Mean accuracy | Minimum seed |
| ------------------ | -----------------: | -----------: |
| Joint CompactionRL |        **99.693%** |  **99.669%** |
| Frozen summary     |            51.815% |            — |
| Joint minus frozen | **+47.878 points** |            — |

The preregistered gate required:

- joint mean accuracy at least 95%;
- every joint seed at least 90%;
- joint minus frozen at least 30 percentage points.

All three gates passed. This demonstrates the causal idea in a controlled discrete policy: training the summary channel can preserve information across repeated context resets when a fixed summary cannot.

It does not demonstrate that a pretrained language model improves on SWE-bench. That requires a different experiment.

---

## 4. Integrating With Public slime

I pinned public slime at commit [`fb42ae4`](https://github.com/THUDM/slime/tree/fb42ae456fac8166afb604f13b30d22bb3c75053). The stack already exposes most required primitives:

- PPO with a critic;
- global per-token loss;
- a custom advantage function hook;
- custom rollout generation;
- `list[Sample]` fan-out from one episode;
- shared rollout IDs;
- response loss masks;
- sampled token IDs and behavior log probabilities.

The reproduction adds two out-of-tree pieces:

1. a custom rollout that emits chronological execution and summary siblings;
2. an advantage hook that restores sibling order and applies cross-trajectory GAE.

This is a better first integration than a large framework fork. The algorithm remains testable without GPUs, while slime owns distributed rollout, Megatron training, SGLang serving, and checkpointing.

### The DP=1 constraint

The current hook must see every sibling from one logical rollout at once. Public slime can partition siblings across data-parallel ranks before the custom advantage function runs.

The safe out-of-tree configuration therefore uses data-parallel size 1 for the actor and critic. A future upstream improvement should either:

- keep all siblings from one rollout on the same DP rank; or
- compute custom advantages before DP partitioning.

Transporting explicit `segment_kind` and `segment_order` metadata into the training batch would also remove the need to encode chronology through `Sample.index`.

---

## 5. The Live Four-A10 Experiment

The live run used:

- model: `Qwen/Qwen2.5-0.5B-Instruct`;
- slime commit: `fb42ae456fac8166afb604f13b30d22bb3c75053`;
- actor: 1 A10;
- critic: 1 A10;
- rollout: 2 A10s across two SGLang engines;
- rollout batch: 4;
- one PPO update;
- debug rollout, training tensors, actor/critic gradient norms, and checkpoint enabled.

The GPU job completed in 354.5 seconds. A separate CPU process then audited the saved artifacts without calling the training hook again.

The publication gate required more than a zero exit code:

- a real execution-summary-execution path;
- shared logical rollout identity and terminal reward;
- summary tokens enabled for loss;
- environment observations masked from loss;
- independent GAE and return recomputation within `5e-5`;
- finite nonzero actor and critic gradients;
- a post-update checkpoint;
- hashes and sizes for the rollout, training, gradient, and checkpoint evidence.

The final result was:

| Live measurement                           |      Result |
| ------------------------------------------ | ----------: |
| Logical rollouts                           |           4 |
| Trainable samples                          |          16 |
| Compact logical rollouts                   |           3 |
| Complete execution-summary-execution paths |           3 |
| Summary segments                           |           6 |
| Execution segments                         |           9 |
| Independently audited train groups         |           4 |
| Maximum advantage error                    |   `6.98e-6` |
| Maximum return error                       |   `6.98e-6` |
| Actor gradient norm                        |   `43.0648` |
| Critic gradient norm                       | `3825.2558` |
| Checkpoint files                           |          11 |
| Checkpoint tree size                       |     6.92 GB |
| Positive trainable-segment rewards         |  **0 / 16** |

All 11 engineering and numerical checks passed.

The last row matters. The unadapted 0.5B instruct model did not follow the console command protocol in this batch. Nonzero policy gradients came from the critic-based advantage structure even though all four logical tasks had zero terminal reward. This live run proves that the compact rollout, critic, actor, loss masks, cross-segment advantages, gradient flow, and checkpoint path are wired correctly. It does **not** prove model capability improvement.

That is why the repository keeps two evidence layers:

```text
96-step multi-seed experiment
    → causal evidence that trainable summaries can solve the long horizon

live Qwen/slime one-update run
    → engineering and numerical evidence that the public-stack integration works
```

Combining them is informative. Conflating them would be misleading.

---

## 6. Five Integration Bugs That Were Worth Finding

The final run passed only after several failures that local unit tests could not fully expose.

### 1. A shell apostrophe changed parsing without failing `bash -n`

An error message contained `slime's` inside a parameter expansion. The unmatched interaction with a later quoted Python snippet caused Bash to absorb a large part of the launcher, eventually surfacing as an unrelated unbound `MASTER_ADDR`.

Lesson: for generated launchers, use `bash -x` with a fake Ray executable and inspect the fully expanded final command. Syntax-only validation is not enough.

### 2. Ray did not inherit the shell's working directory

The launcher ran `cd /root/slime`, then submitted `python train.py`. The Ray job started in `/root` and failed because `/root/train.py` did not exist.

Fix: submit the absolute `/root/slime/train.py` path.

### 3. The repository root was importable, but `src/` was not

Ray workers could import `integrations.slime.rollout`, then failed on `compactionrl.simulation` because the package lives under `src/`.

Fix: include both the repo root and `repo/src` in the Ray runtime `PYTHONPATH`.

### 4. slime expects a dotted function path

The first hook path used Python packaging's familiar `module:function` form. slime's loader calls `rpartition(".")`, so it requires:

```text
compactionrl.adapters.slime.compactionrl_advantages
```

This failed only after rollout generation and critic training, immediately before actor advantage computation.

### 5. `TRUNCATED` samples are still trainable

The first independent audit correctly counted only `COMPLETED` samples for completion statistics, but then incorrectly used that same subset to join training metadata. slime also trains valid `TRUNCATED` responses.

Fix: keep completion statistics and trainable-sample auditing as separate sets. After the fix, all four groups—not only the compact completed group—matched the independent GAE calculation.

These bugs are part of the result. A framework integration is not complete when the adapter imports. It is complete when real saved tensors reconstruct the intended algorithm after distributed packing and training.

---

## 7. Reproduce the Reproduction

The framework-neutral checks need no third-party package:

```bash
git clone https://github.com/ajing/compactionrl-repro.git
cd compactionrl-repro
bash scripts/run_checks.sh
```

That command runs 40 algorithm and evidence-gate tests, regenerates the toy actor-critic result, and reruns the 96-step long-horizon experiment.

To validate against a pinned slime checkout:

```bash
python3 scripts/check_slime_compat.py --slime-root /path/to/slime
```

The Modal path is explicit about spend:

```bash
uvx --from modal modal run integrations/slime/modal_e2e.py \
  --mode prepare --confirm spend_modal_credits

uvx --from modal modal run integrations/slime/modal_e2e.py \
  --mode one-batch --confirm spend_modal_credits
```

The one-batch command automatically launches the independent audit. The repository's final report builder refuses to publish a PASS if the live audit, actor gradient, critic gradient, compaction path, GAE parity, or checkpoint evidence is missing.

---

## 8. What I Would Do Next

The next useful experiment is not to rerun the same zero-reward 0.5B batch many times. It is to make the small-model task learnable while preserving the long-horizon causal structure.

I would stage it this way:

1. add a constrained command grammar or a short supervised warm start so the model can produce `next` and `submit`;
2. run several PPO updates until terminal rewards become nonzero;
3. compare full CompactionRL against frozen-summary, no token normalization, and no cross-trajectory correction;
4. report learning curves and multiple seeds, not one final checkpoint;
5. only then move to a small SWE-Dev slice with an executable verifier;
6. upstream segment metadata and whole-rollout DP placement into slime before scaling.

Paper-scale work should begin with the disclosed 30B setting, frozen dataset IDs, prompts, tokenizer, scaffold, and container commits. The 106B or internal GLM-5.2 regime should come only after the smaller ablations reproduce directionally.

---

## Bottom Line

CompactionRL's important contribution is not "summarize when the context is full."

It is the combination of:

- making the summary a policy action;
- preserving exact token provenance after context rewriting;
- sharing terminal reward across execution and summary siblings;
- correcting cross-segment temporal credit;
- normalizing PPO over enabled tokens rather than segments.

The controlled experiment shows that this can solve a genuine long-horizon information bottleneck. The live slime run shows that the open implementation reaches real rollout engines, actor and critic gradients, an independently verified advantage calculation, and a checkpoint.

The remaining gap to the paper's headline results is no longer a missing algorithm skeleton. It is model capability, a learnable task interface, undisclosed recipe details, evaluation scale, and compute.

Code: [`ajing/compactionrl-repro`](https://github.com/ajing/compactionrl-repro)  
Evidence: [final E2E report](https://github.com/ajing/compactionrl-repro/blob/main/results/final_e2e_report.md)  
Algorithm source: [CompactionRL](https://arxiv.org/abs/2607.05378)  
Related model sources: [GLM-5 technical report](https://arxiv.org/abs/2602.15763), [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2)
