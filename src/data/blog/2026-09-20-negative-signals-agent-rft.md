---
author: Jing Lu
pubDatetime: 2026-09-20T18:00:00-07:00
title: "Can RFT Learn From Failure? Negative Signals for Agent Fine-Tuning"
featured: true
draft: false
tags:
  - AI
  - LLM Agents
  - Post Training
  - Reinforcement Learning
  - Evaluation
description: "A technical guide to learning from failed agent rollouts through step masking, unlikelihood, DPO, SimPO, and step-level credit assignment."
---

Agent training often throws away its most expensive data.

An agent samples many rollouts. A verifier scores each final result. The training pipeline keeps successful rollouts and discards failed ones. It then uses supervised fine-tuning, or SFT, on the accepted data.

This method learns from success. It does not learn what to avoid. A failed rollout gets zero gradient because it never enters the loss.

We can add a negative signal without an online reinforcement learning loop. The data pipeline can still use offline rollouts and teacher forcing. The loss must change, so the method is no longer plain SFT.

For long agent tasks, the main question is not whether to use failed rollouts. The main question is where the failure started.

## What RFT means in this post

The term RFT has several meanings. This post uses the definition from recent agent research: rejection fine-tuning.

The pipeline has four steps:

```text
sample rollouts
      |
score final outcomes
      |
keep successful rollouts
      |
run supervised fine-tuning
```

Some papers call this rejection sampling fine-tuning. It is not online policy-gradient training. The final optimization step is behavior cloning on accepted actions.

Let $s_t$ be the agent state before step $t$. The state contains the prompt, prior actions, and tool results. Let $a_t$ be the action that the model produced.

The positive-only loss is:

$$
\mathcal{L}_{\text{SFT}}
=
-\sum_{(s_t,a_t)\in D^+}
\log \pi_\theta(a_t\mid s_t)
$$

The dataset $D^+$ contains accepted steps. A rejected action does not appear in this sum. Its probability does not move down through an explicit training signal.

## A warning in the prompt is not a negative gradient

Suppose the SFT target says:

> Do not delete the database.

The model learns to produce that sentence in a similar context. This target does not directly reduce the probability of a database deletion action.

A true negative signal must enter the loss. It must lower the probability of a specific action, span, or trajectory.

This difference matters for agents. Text about an error and a gradient against the error are different training signals.

## Why a signed cross-entropy loss is unsafe

A direct idea uses an advantage value $A$ as a weight:

$$
\mathcal{L}(s,a)
=
-A(s,a)\log\pi_\theta(a\mid s)
$$

A positive advantage raises the action probability. A negative advantage appears to lower it.

The negative case has a problem. If $A<0$, then:

$$
\mathcal{L}(s,a)
=
|A|\log\pi_\theta(a\mid s)
$$

As $\pi_\theta(a\mid s)$ approaches zero, the loss approaches negative infinity. The objective has no lower bound. Training can push the rejected action down without a natural stopping point.

Online policy-gradient methods control this behavior through sampling, baselines, clipping, and policy constraints. A static SFT batch does not provide those controls by itself.

This fact explains a common practice. Advantage-weighted SFT uses nonnegative weights such as:

$$
w(s,a)=\exp\left(\frac{A(s,a)}{\beta}\right)
$$

A bad action gets a small weight instead of a negative weight. This method changes which actions the model imitates. It does not directly suppress the bad action.

## Five ways to use failed rollouts

Failed data can affect training at five different levels.

| Method                 | Uses failed data | Explicit negative gradient | Main unit             |
| ---------------------- | ---------------: | -------------------------: | --------------------- |
| Rejection fine-tuning  |               No |                         No | Successful trajectory |
| Step masking           |              Yes |                         No | Accepted step         |
| Advantage-weighted SFT |              Yes |                         No | Weighted step         |
| Unlikelihood           |              Yes |                        Yes | Bad token or action   |
| DPO, SimPO, or KTO     |              Yes |                        Yes | Preferred comparison  |

These methods do not make the same claim. Step masking saves useful parts of failures. Preference training also pushes a rejected choice down.

## Step masking uses failure without negative learning

[Step Rejection Fine-Tuning](https://arxiv.org/abs/2605.10674), or SRFT, asks a critic to label each step. It masks the loss on steps that the critic marks as wrong. The wrong step remains in the input context.

This design creates a useful pattern:

```text
good action       -> SFT loss
good action       -> SFT loss
bad action        -> no loss, keep in context
error observation -> keep in context
recovery action   -> SFT loss
```

The model can learn how to recover after an error. It does not learn to reproduce the error.

SRFT reported a `32.2%` resolution rate on SWE-bench Verified. The RFT baseline reached `30.9%`. This result supports step filtering, but SRFT still has no explicit gradient against the bad action.

That distinction is important. Masking means “do not imitate.” A negative objective means “reduce this probability.”

## Unlikelihood gives a direct negative signal

[Unlikelihood training](https://arxiv.org/abs/1908.04319) adds a bounded loss for a known bad action:

$$
\mathcal{L}_{\text{UL}}
=
-\log\left(1-\pi_\theta(a^-\mid s)\right)
$$

The loss approaches zero when the bad action probability approaches zero. It grows without bound when that probability approaches one.

For the selected action logit, the gradient magnitude is its current probability. A bad action gets a stronger correction when the model assigns more probability to it.

This method works well when the bad action is clear:

- The tool name does not exist.
- The JSON arguments fail schema rules.
- The action breaks a known safety rule.
- A test identifies the first harmful edit.

Unlikelihood is harder for long free-form actions. A bad action can have many token forms. Reducing one string can move probability to a similar string with the same effect.

The safest unit is often the first wrong action span. Token-level labels inside a long explanation can become arbitrary.

## Pairwise training asks a better question

Preference training compares two choices from the same input. The data contains a preferred action $a^+$ and a rejected action $a^-$.

The target relation is:

$$
\log\pi_\theta(a^+\mid s)
>
\log\pi_\theta(a^-\mid s)
$$

This relation does not require an absolute label for every possible action. It only needs a reliable comparison.

### DPO keeps a reference policy

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290), or DPO, compares policy changes against a fixed reference model:

$$
\begin{aligned}
\Delta_\theta
=
&\log\frac{\pi_\theta(a^+\mid s)}
{\pi_{\text{ref}}(a^+\mid s)} \\
&-
\log\frac{\pi_\theta(a^-\mid s)}
{\pi_{\text{ref}}(a^-\mid s)}
\end{aligned}
$$

The loss is:

$$
\mathcal{L}_{\text{DPO}}
=
-\log\sigma\left(\beta\Delta_\theta\right)
$$

The reference term anchors the update to the starting policy. The loss learns relative change instead of ranking raw policy scores alone.

DPO uses the same basic training stack as SFT:

```text
offline examples
-> teacher-forced forward passes
-> token log probabilities
-> loss
-> backward pass
```

It needs two policy sequences and two reference sequences for each pair. This cost can matter for long agent contexts.

### SimPO removes the reference model

[Simple Preference Optimization](https://arxiv.org/abs/2405.14734), or SimPO, uses average sequence log probability as its reward:

$$
r_\theta(s,a)
=
\frac{\beta}{|a|}
\log\pi_\theta(a\mid s)
$$

It adds a target margin $\gamma$:

$$
\mathcal{L}_{\text{SimPO}}
=
-\log\sigma
\left(
r_\theta(s,a^+)
-r_\theta(s,a^-)
-\gamma
\right)
$$

SimPO does not load a reference model. Length normalization also reduces the built-in penalty on longer sequences.

The original SimPO experiments studied chat responses. Applying the same loss to tool-action spans is a new design choice. Agent training must test summed and length-normalized scores separately.

A long invalid tool call can contain many harmless structure tokens. Length normalization can weaken the effect of the actual error.

### KTO works without matched pairs

[KTO](https://arxiv.org/abs/2402.01306) accepts separate desirable and undesirable examples. It does not require a chosen and rejected output for every prompt.

This format fits rollout stores with binary rewards and few matched branches. It gives less control over prompt difficulty. Easy successes and hard failures can create a shortcut instead of a useful action preference.

## Whole-trajectory preference has a credit problem

A failed trajectory is rarely wrong at every step.

Consider this coding-agent rollout:

```text
1. inspect the repository       correct
2. find the failing test        correct
3. identify the target file     correct
4. edit the wrong function      wrong
5. run tests                    useful
6. fail the task                final outcome
```

The trajectory log probability is a sum:

$$
\log\pi_\theta(\tau)
=
\sum_t
\log\pi_\theta(a_t\mid s_t)
$$

A sequence-level rejected loss pushes the full sum down. The first three correct actions receive the same rejection sign as the bad edit.

There is a second problem. Two trajectories share a state only before their first different action. After that action, the tools return different observations. The later actions happen in different states.

A comparison between later actions can therefore mix two effects:

- Action quality.
- State difficulty after an earlier choice.

The clean comparison occurs at the first branch:

```text
same state s_t
├── action a+ -> higher verified return
└── action a- -> lower verified return
```

This structure turns trajectory data into a local policy comparison.

## Step-level credit is the central problem

Terminal reward says whether the task finished. It does not identify the action that caused the result.

Agent training needs a step score such as an advantage:

$$
A_t
=
Q(s_t,a_t)-V(s_t)
$$

$Q(s_t,a_t)$ estimates the future return after one action. $V(s_t)$ estimates the expected return before that action.

An agent pipeline can estimate step value from several sources:

1. Run sibling actions from the same saved state.
2. Use executable tests after a code edit.
3. Ask a calibrated critic to identify the first wrong step.
4. Retry from each prefix and measure later success.
5. Compare a failed path with a corrected path that shares its prefix.

The strongest label uses the same state and an executable outcome. A critic-only label is cheaper, but critic errors become training errors.

Do not put future evidence in the model input. A future test result can label an earlier action. The policy must still predict that action from the information available at the time.

## Failed trajectories contain three kinds of data

A failed rollout is not one training example. It contains three different assets.

### Useful actions before the error

The model can imitate correct plans, searches, and tool calls. [Exploring Expert Failures](https://arxiv.org/abs/2504.13145) follows this idea. It extracts beneficial actions and removes harmful ones.

The method reached a `62%` win rate on WebShop. Its RFT baseline reached `53.6%`. This comparison shows that failure mining can add useful coverage.

### The harmful branch action

The first harmful action is a candidate for unlikelihood or a pairwise loss. The label needs high precision because a false negative trains the model against a valid choice.

### Recovery after the error

Later actions can teach recovery. Keep the bad action and its tool result in the context. Mask the loss on the bad action, then supervise the useful recovery action.

This pattern trains the state distribution that the student will visit after its own mistakes.

## Critical steps can matter more than full imitation

[ATLaS](https://arxiv.org/abs/2503.02197) selects planning, reasoning, and decision steps from expert trajectories. Its model used about `30%` of the steps and outperformed full-trajectory fine-tuning.

This result does not prove that full trajectories always hurt. It shows that supervised tokens have unequal value.

Routine formatting tokens can dominate a long trajectory. A small number of decisions can determine task success. Equal loss on every token can spend most of the update on protocol imitation.

## A practical loss for agent rollout distillation

A useful first system combines three objectives:

$$
\mathcal{L}
=
\mathcal{L}_{\text{positive}}
+\lambda_{\text{pair}}\mathcal{L}_{\text{step-pair}}
+\lambda_{\text{UL}}\mathcal{L}_{\text{invalid}}
$$

The positive loss imitates verified good actions and recovery actions. The pair loss compares two actions from the same state. The unlikelihood loss covers clear invalid actions.

Do not apply every negative loss to every failure. Use this rule:

- Use SFT when an action is clearly good.
- Use masking when the label is uncertain.
- Use pairwise loss when two actions share a state.
- Use unlikelihood when an action is clearly invalid.
- Use online RL when the policy must discover new states.

For a first experiment, use positive SFT plus step-level DPO. Add unlikelihood only for deterministic errors such as invalid tool calls.

## Data construction matters more than the loss name

The following pipeline preserves the useful information in agent rollouts:

```text
rollouts
   |
segment model actions and tool results
   |
save the state before each action
   |
label good, bad, and unknown steps
   |
build same-state action pairs
   |
keep recovery contexts
   |
train positive and negative objectives
```

Mask all environment observation tokens from the target loss. The tool produced those tokens, not the policy.

Also mask user messages, system messages, and hidden grader text. Train only the model outputs that the deployed policy must produce.

Pair actions inside the same task and state when possible. A success from an easy prompt and a failure from a hard prompt do not form a clean preference pair.

## The experiment that can prove value

A base-to-SFT comparison is not enough. It cannot separate policy learning from format learning.

Use one rollout pool and train these models:

| Model | Training data and loss                                   |
| ----- | -------------------------------------------------------- |
| M0    | Base model                                               |
| M1    | Successful trajectories with SFT                         |
| M2    | Successes and full failed trajectories with SFT          |
| M3    | Successes plus failed trajectories with bad steps masked |
| M4    | Success and failure pairs with trajectory-level SimPO    |
| M5    | Same-state action pairs with step-level DPO              |
| M6    | Positive SFT plus unlikelihood on invalid actions        |
| M7    | Online RL from the same starting model                   |

Keep these factors fixed:

- The base checkpoint.
- The agent scaffold and prompts.
- The tools and tool versions.
- The rollout pool for offline methods.
- The inference step and token limits.
- The decoding temperature and sample count.

Report both supervised tokens and environment interactions. A token-matched experiment and an interaction-matched experiment answer different cost questions.

Use several training seeds. Evaluate each model on the same task instances. Use paired bootstrap intervals across tasks.

## Measure the policy, not only the final score

Task success is necessary, but it does not show why the model improved.

Use these diagnostics:

### Fixed-prefix action margin

Give every model the same state $s_t$. Measure:

$$
M(s_t)
=
\log\pi(a^+\mid s_t)
-
\log\pi(a^-\mid s_t)
$$

This test measures the local policy change without rollout noise.

### First-error position

Record the first verified error in each rollout. A better policy must move this error later or remove it.

### Recovery from a counterfactual state

Insert a known bad action into an otherwise valid prefix. Give the resulting observation to the model. Measure whether it recovers.

Positive-only RFT often lacks these states. SRFT-style data can train them directly.

### Invalid tool rate

Count unknown tools, schema failures, missing arguments, and actions that the environment rejects. This metric isolates protocol learning.

### Generalization by distance

Evaluate separate groups:

1. New instances in a known environment.
2. New tools with different schemas.
3. New environments with the same task type.
4. Longer tasks than the training horizon.
5. Hard tasks that the teacher rarely solves.

[WebLINX](https://arxiv.org/abs/2402.05930) shows why this split matters. Fine-tuned models performed well on familiar websites but struggled on unseen websites.

### Capability retention

Measure general instruction following, coding, and reasoning after agent training. A narrow negative loss can damage useful behaviors that share the same tokens.

Also report KL drift from the base model. Drift is not a quality metric, but it helps explain broad regressions.

## What existing results establish

The literature supports several parts of this design.

[FireAct](https://arxiv.org/abs/2310.05915) showed that 500 GPT-4 agent trajectories improved Llama 2 7B HotpotQA performance by `77%`. Agent SFT can transfer tool-use patterns from a stronger teacher.

[SWE-Gym](https://arxiv.org/abs/2412.21139) reported gains of up to `19` absolute points on SWE-bench Verified and Lite. Its executable tests provide stronger outcome labels than a text-only judge.

ATLaS shows that critical-step selection can beat full imitation. EEF shows that failed trajectories can contain useful actions. SRFT shows that a model can learn recovery while bad steps stay masked.

These papers do not yet prove that step-level DPO is the best agent objective. They establish the data problem that such an objective tries to solve.

## When each method fits

Use positive-only RFT when the verifier is reliable and successful rollouts are plentiful. This method is simple and stable.

Use step masking when failed rollouts contain useful work but step labels remain uncertain. Masking limits damage from critic errors.

Use unlikelihood for actions with an objective failure rule. Invalid tools and unsafe commands fit this case.

Use DPO when matched alternatives share the same state and policy anchoring matters. Use SimPO when reference-model cost is a larger concern.

Use KTO when labels are binary and matched pairs are rare. Control for prompt difficulty because desirable and undesirable samples can come from different state distributions.

Use online RL when new behavior requires exploration. Offline losses cannot learn actions that never appear in the rollout store.

## The main answer

Negative signals can fit inside the SFT training infrastructure. The optimizer still reads offline data, computes token log probabilities, and runs backpropagation.

The objective is no longer vanilla SFT. It becomes unlikelihood, preference optimization, or another offline policy objective.

For long-horizon agents, whole-trajectory rejection is too coarse. The best training unit is often a state and two competing actions.

The practical target is:

$$
\text{imitate good actions}
+
\text{suppress verified bad actions}
+
\text{learn recovery after errors}
$$

This design keeps more information from RL rollouts. It also respects the central fact of agent training: a failed trajectory can contain good decisions, one harmful branch, and a useful recovery attempt.

## References

- Sean Welleck et al. [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319), ICLR 2020.
- Rafael Rafailov et al. [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), NeurIPS 2023.
- Baian Chen et al. [FireAct: Toward Language Agent Fine-Tuning](https://arxiv.org/abs/2310.05915), 2023.
- Kawin Ethayarajh et al. [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306), 2024.
- Xing Han Lù, Zdeněk Kasner, and Siva Reddy. [WebLINX: Real-World Website Navigation with Multi-Turn Dialogue](https://arxiv.org/abs/2402.05930), 2024.
- Yu Meng, Mengzhou Xia, and Danqi Chen. [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734), NeurIPS 2024.
- Jiayi Pan et al. [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139), 2024.
- Zhixun Chen et al. [ATLaS: Agent Tuning via Learning Critical Steps](https://arxiv.org/abs/2503.02197), 2025.
- Li-Cheng Lan et al. [Exploring Expert Failures Improves LLM Agent Tuning](https://arxiv.org/abs/2504.13145), 2025.
- Igor Slinko et al. [Step Rejection Fine-Tuning: A Practical Distillation Recipe](https://arxiv.org/abs/2605.10674), 2026.
