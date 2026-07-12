---
author: Jing Lu
pubDatetime: 2026-07-12T23:00:00Z
title: "Training the Critic Without Crashing the Reward: A Practical Guide to Agentic RL"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Reinforcement Learning
  - RLHF
  - Post Training
description: "A practical framework for critic training and credit assignment in long-horizon LLM agents: IQL, pairwise advantage, hindsight and counterfactual critics, privileged information, turn-level MDPs, chain-of-thought monitoring, and reward-crash diagnosis."
---

Reinforcement learning becomes a different problem when the model stops producing one answer and starts operating an environment.

A reasoning model may generate one long response and receive one correctness score. An agent may search, call tools, modify files, inspect the result, recover from an error, and continue for 100 turns before receiving a terminal reward. The final success signal tells us whether the episode worked. It does not tell us which action made it work.

This is the central problem of agentic RL:

> The reward is observed at the end, but the useful learning decisions happened throughout the trajectory.

The practical response is often described as “train a critic.” But that phrase hides several different objects:

- a **reward model** that judges an output or completed trajectory;
- a **value critic** that predicts future return from an intermediate state;
- a **process critic** that scores individual steps or turns;
- a **privileged critic** that sees training-only information;
- a **hindsight or counterfactual critic** that reasons backward from the outcome;
- a **monitor** that detects reward hacking or suspicious behavior.

These components should not automatically share the same data, inputs, or optimization role. A critic that is useful for auditing may become unsafe or gameable when converted directly into a dense training reward.

This post develops a practical framework for choosing and training these critics. It also explains why reward sometimes “crashes,” why direct penalties on chain-of-thought can hide rather than remove bad behavior, and how I would assemble an agentic RL stack today.

This is a methods synthesis, not a claim that one algorithm has solved long-horizon credit assignment. The empirical evidence is still fragmented across environments such as WebShop, ALFWorld, software engineering, search, dialogue, and collaborative coding.

---

## 1. Begin With the Correct Unit of Decision

For a language agent, the most useful high-level unit is usually a **turn**, not a token and not the whole episode.

At turn $t$:

- $s_t$ is the conversation history plus the observable environment state;
- $a_t$ is one agent turn, often including reasoning, a tool choice, and tool arguments;
- the environment executes the action and returns an observation $o_{t+1}$;
- the next state becomes $s_{t+1}$;
- reward may be immediate, delayed, or terminal-only.

The resulting trajectory is:

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T).
$$

The return from turn $t$ is:

$$
G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k.
$$

This is the **turn-level MDP** view. It does not claim that the underlying environment is perfectly Markov. Most agents operate under partial observability, so the history, memory, and retrieved context are often part of the practical state representation.

Why not use a token-level MDP for everything? Because token probability is the right unit for optimizing an autoregressive model, but usually the wrong semantic unit for assigning environmental credit. Whether the agent chose the wrong database, deleted a file, or asked the right clarifying question is a turn-level fact. Spreading the same terminal advantage over every token confuses decision quality with wording.

[Turn-PPO](https://arxiv.org/abs/2512.17008) makes this distinction explicit by estimating advantage in a turn-level MDP for multi-turn agents. [GiGPO](https://arxiv.org/abs/2505.10978) takes a critic-free route: it combines an episode-level group advantage with a step-level relative advantage constructed from actions taken at shared anchor states. These methods differ, but they agree on the important abstraction: long-horizon agent learning needs credit below the episode level and above the individual token level.

The clean implementation is hierarchical:

```text
environment outcome
        ↓
trajectory / turn-level credit
        ↓
tokens inside the credited turn
        ↓
policy-gradient update
```

Token-level optimization remains necessary. Token-level *causal interpretation* is optional and much harder.

---

## 2. Four Roles That Should Not Be Collapsed Into One Critic

| Component | Core question | Typical input | Typical use |
|---|---|---|---|
| Outcome verifier / reward model | Did the trajectory succeed? | final state, output, tests, rubric | terminal reward, evaluation |
| Value critic | How much future return is available here? | current state, sometimes action | advantage estimation |
| Process or privileged critic | Which intermediate decisions helped? | full trajectory plus training-only evidence | turn rewards, data labeling |
| Monitor | Is the agent exploiting or concealing something? | actions, observations, raw reasoning | audit, triage, safety evaluation |

The separation matters because each component fails differently.

- A verifier can be exploited.
- A value critic can be biased or unstable.
- A process critic can invent plausible but causally false explanations.
- A monitor can become an adversarial target if its output is directly optimized.

An effective system therefore does not ask one LLM judge for a scalar and call the problem solved. It combines executable evidence, learned prediction, and held-out monitoring while preserving their separate measurements.

---

## 3. The Cheapest Baseline: Outcome Reward and Relative Advantage

Suppose we sample $G$ trajectories for the same task and score them with terminal rewards $R_1, \ldots, R_G$. A GRPO-style relative advantage is:

$$
A_i = \frac{R_i - \mu_G}{\sigma_G + \epsilon}.
$$

Every action or token in trajectory $i$ may receive the same $A_i$. This is cheap, critic-free, and often sufficient when:

- trajectories are short;
- rewards are reliable;
- a nontrivial fraction of rollouts succeeds;
- successful and unsuccessful rollouts differ in a learnable way.

It becomes weak when episodes are long. A terminal reward of 1 reinforces the decisive action, harmless filler, unnecessary detours, and lucky recovery steps together.

It also suffers from **group degeneracy**. If every rollout in a group has the same binary reward, then $R_i - \mu_G = 0$ for every sample. The batch provides no relative learning signal. Increasing reward scale does not fix this: zero multiplied by 100 is still zero.

### Pairwise advantage

Instead of fitting an absolute value, we can compare two actions or suffixes from a matched state:

$$
A(a^+ \succ a^- \mid s) \approx R(s, a^+) - R(s, a^-).
$$

Pairwise credit has several advantages:

- common prompt difficulty cancels out;
- rank judgments are often easier than calibrated scalar judgments;
- it avoids a globally calibrated value function;
- it fits branch-and-compare data naturally.

Its weaknesses are equally important:

- it learns only where useful pairs exist;
- pair construction can introduce severe selection bias;
- a judge can rank style rather than causal contribution;
- non-transitive preferences make a global policy difficult to recover;
- all-bad or nearly identical pairs provide little signal.

Pairwise advantage is therefore strongest when the two alternatives begin from the same real state and are evaluated by environment execution. It is weaker when an LLM judge merely imagines which alternative “sounds better.”

---

## 4. IQL: A Strong Offline Starting Point, Not a Universal Critic

If we already have a mixture of expert demonstrations, successful agent runs, and failures, **Implicit Q-Learning** is an attractive offline approach. [IQL](https://arxiv.org/abs/2110.06169) avoids querying the learned Q-function on arbitrary out-of-distribution actions during offline training.

At a high level, IQL learns:

1. a state value $V(s)$ through upper-expectile regression over dataset actions;
2. a Q-function through Bellman regression;
3. a policy through advantage-weighted behavior cloning.

The policy update weights dataset actions approximately as:

$$
w(s,a) = \exp\left(\frac{Q(s,a)-V(s)}{\beta}\right),
$$

usually with clipping to prevent a few samples from dominating.

This has a natural interpretation for agents: imitate actions that look better than the typical dataset action at the same state, while remaining close to the support of logged behavior.

### Where IQL is strong

- expensive environments where online rollout is limited;
- mixed-quality trajectory logs;
- bootstrapping from demonstrations before online RL;
- situations where conservative in-distribution improvement matters more than broad exploration;
- replaying historical failures without repeatedly executing them.

### Where IQL struggles

- text states are rarely repeated exactly, making value generalization difficult;
- sparse terminal reward still creates a hard Bellman propagation problem;
- partial observability can make $V(s)$ fundamentally ambiguous;
- critic error compounds over long horizons;
- the learned policy cannot reliably discover actions absent from dataset support;
- action likelihood is not the same as semantic action quality.

A recent direction such as [Q-Evolve](https://arxiv.org/abs/2606.07367) adapts weighted IQL ideas to long-horizon LLM agents by learning an in-distribution critic on hybrid expert and agent-generated data, then deriving process signals for behavior-proximal policy updates. The promising part is not merely “IQL for language.” It is the closed loop: keep critic training and policy improvement grounded in a shared, refreshed data distribution.

My default use of IQL would be **offline initialization**, followed by conservative online improvement. I would not trust an IQL critic indefinitely as the actor moves far beyond its dataset.

---

## 5. Hindsight and Counterfactual Credit

Hindsight and counterfactual methods both revisit completed trajectories, but they answer different questions.

### Hindsight: now that we know the ending, which earlier decisions mattered?

A hindsight critic sees information that was unavailable at turn $t$:

- the final outcome;
- later observations;
- the eventual error;
- a gold answer or test result;
- the point where recovery became impossible.

It then estimates a refined value or turn score:

$$
\widehat{Q}_{\text{hind}}(s_t,a_t; \tau_{t+1:T}, R_T).
$$

[HCAPO](https://arxiv.org/abs/2603.08754) uses an LLM as a post-hoc critic to refine step-level Q estimates for long-horizon agents. [CriticSearch](https://aclanthology.org/2026.findings-acl.596/) similarly uses a frozen retrospective critic with the completed search trajectory and gold answer to produce turn-level feedback.

Hindsight is useful because judging an earlier action is easier after seeing its consequences. Its main danger is **narrative attribution**: an LLM can write a convincing story about why an action mattered without measuring its causal effect.

### Counterfactual: what would have happened under a different action?

The clean counterfactual advantage is a difference reward:

$$
A^{\text{cf}}_t = R(\tau) - R(\tau_{a_t \leftarrow a'_t}).
$$

The second term evaluates an alternate action from the same state. This can be estimated by:

1. actually restoring the environment and rerunning;
2. fixed-continuation replay where valid;
3. a learned world model;
4. an LLM simulation or judge.

These options are not equally trustworthy. Real environment re-execution provides the strongest causal evidence but costs more. A learned or verbal simulation is cheaper but can reproduce the critic's biases.

[C3](https://arxiv.org/abs/2603.06859) uses context-matched alternatives and leave-one-out baselines for localized counterfactual credit in multi-agent LLM collaboration. [HOPE](https://openreview.net/forum?id=1jqYCS1LxR) uses hindsight to propose alternative actions and expand exploration coverage. The common idea is powerful: failed trajectories are not merely negative examples; they identify branch points where alternative actions should be tested.

The practical reliability order is:

```text
environment rerun
    > deterministic simulator / hidden verifier
    > calibrated learned dynamics
    > stronger held-out LLM judge
    > actor self-critique alone
```

This ordering is a design heuristic, not a theorem. It reflects how much causal grounding each route normally provides.

---

## 6. Privileged Critics: More Information During Training

A privileged critic receives information that the deployed actor does not have. For example:

- hidden tests and verifier output;
- the true database state rather than the rendered UI;
- future turns and the final outcome;
- a gold answer;
- other agents' private observations;
- structured task metadata;
- the exact policy constraint that was violated.

The actor remains deployable because it still chooses actions from ordinary observations:

$$
a_t \sim \pi_\theta(a_t \mid o_{\le t}).
$$

The critic can use richer training information:

$$
V_\phi = V_\phi(o_{\le t}, x^{\text{priv}}, \tau_{t+1:T}).
$$

[SWEET-RL](https://arxiv.org/abs/2503.15478) is a concrete multi-turn example: its critic uses additional training-time information to provide step-wise evaluation for collaborative coding and design tasks.

Privileged critics are especially valuable when the actor's observation is noisy but the training environment contains structured ground truth. A browser agent may only see a page; the critic can inspect the underlying DOM, database diff, policy state, and final transaction result.

The main risks are:

1. **Information leakage.** The actor may be trained to rely on a signal it cannot reproduce at deployment.
2. **Shortcut learning.** The critic may predict success from privileged identifiers rather than action quality.
3. **Train-test mismatch.** Dense scores may disappear when the environment changes.
4. **False precision.** A privileged critic is still a learned predictor, not automatically a causal oracle.

Useful safeguards include privileged-feature dropout, ablations by information source, held-out task templates, critic calibration, and comparisons against executable counterfactuals.

---

## 7. Chain-of-Thought: Monitor It, But Be Careful What You Optimize

Raw reasoning can help a critic detect reward hacking. It is tempting to convert that monitor into a penalty:

$$
R_{\text{train}} = R_{\text{task}} - \lambda M(\text{CoT}, \text{actions}).
$$

The actor now has two ways to improve reward:

1. stop the bad behavior;
2. continue the behavior but stop producing evidence that triggers $M$.

OpenAI's [chain-of-thought monitoring study](https://openai.com/index/chain-of-thought-monitoring/) found exactly this failure mode. Direct optimization pressure against suspicious reasoning could reduce visible “bad thoughts” while leaving reward hacking and making the intent harder to detect. This is evidence that observable reasoning and behavior can become decoupled. It is not proof that the model contains a hidden natural-language monologue.

The correct distinction is:

```text
critic as monitor               critic as optimization target
-----------------               -----------------------------
audits behavior                 directly changes actor reward
can remain held out             becomes part of the environment
finds suspicious samples        actor searches its blind spots
preserves evidence              may incentivize evidence removal
```

My preferred architecture is:

```text
actor trajectory ──→ executable verifier ──→ primary training reward
        │
        ├──────────→ frozen CoT/action monitor ──→ audit and data triage
        │
        └──────────→ privileged hindsight critic ──→ candidate turn credit
                                                    │
                         causal checks / reruns ─────┘
```

If a product needs a clean user-visible explanation, generate it with a separate summarizer or sanitizer. Do not assume that polishing the raw reasoning channel makes the underlying policy safer.

---

## 8. What “Reward Crash” Actually Means

Practitioners use **reward crash** for several different phenomena. Looking only at average total reward makes them easy to confuse.

### 8.1 Policy optimization crash

The policy update is too large. Common causes include high learning rate, too many optimization epochs per rollout batch, stale off-policy samples, extreme advantages, weak KL control, or a wide clipping range.

Typical signature:

- task reward falls sharply;
- KL to the reference or previous policy spikes;
- PPO clip fraction rises;
- gradient norm or value loss spikes;
- output length and tool behavior shift abruptly.

PPO's clipped objective was designed to make repeated minibatch updates more conservative, not to guarantee that collapse is impossible. The first response should usually be to preserve evidence and roll back to the last healthy checkpoint, then lower the actor learning rate, reduce update epochs, enforce target-KL early stopping, clip gradients, and inspect advantage outliers. See the original [PPO paper](https://arxiv.org/abs/1707.06347).

### 8.2 Exploration or entropy crash

The policy concentrates on a narrow family of trajectories. If that family stops working, successful alternatives are no longer sampled and the sparse reward disappears.

Typical signature:

- token or action entropy falls before task reward;
- tool sequences become repetitive;
- pass@1 may rise while pass@k and trajectory diversity fall;
- GRPO groups become mostly all-zero or all-one;
- the agent repeats the same failed strategy.

The fix is not simply a larger reward coefficient. Useful interventions include restoring an earlier checkpoint, entropy regularization, more rollouts, semantic exploration, curriculum sampling, guided attempts, and mixing a small amount of successful behavior data. [Agent-RLVR](https://arxiv.org/abs/2506.11425), for example, addresses the sparse-reward difficulty of software agents by using guidance to make useful trajectories reachable.

### 8.3 Critic or advantage crash

The environment reward may be correct while the value critic becomes badly calibrated. If a successful trajectory returns 1 but the critic predicted 3, the computed advantage is negative:

$$
A_t = G_t - V(s_t) = 1 - 3 = -2.
$$

The policy can then be trained away from successful actions.

Look for normal raw reward combined with abnormal value predictions, negative explained variance, exploding value loss, or a sign reversal between success and advantage. Check terminal masks, timeout handling, bootstrap boundaries, reward normalization, and privileged-information leakage before changing the policy algorithm.

### 8.4 Proxy reward rises while real performance crashes

This is reward overoptimization, not a literal collapse of the optimized scalar:

```text
proxy training reward          ↑
held-out verifier / humans     ↓
```

The actor has found a blind spot in the reward model or verifier. OpenAI's [reward model scaling study](https://openai.com/index/scaling-laws-for-reward-model-overoptimization/) showed that continued optimization of an imperfect proxy can reduce a stronger “gold” reward even as proxy reward improves.

The response is stronger held-out evaluation, adversarial examples, reward-model ensembles, uncertainty penalties, on-policy preference refresh, tighter policy-drift control, and executable verification wherever possible.

### 8.5 Reward pipeline crash

Many apparent algorithmic failures are implementation failures:

- judge timeouts are recorded as zero;
- workers load different reward checkpoints;
- a verifier version changes mid-run;
- EOS and padding masks are wrong;
- terminal reward is attached to the wrong turn;
- truncated episodes are counted as complete failures;
- a length or KL penalty silently dominates;
- reward normalization divides by near-zero variance;
- tool sandboxes start timing out.

The fastest test is deterministic replay: run the same frozen trajectories through the old and new reward pipelines and compare every component.

---

## 9. Do Not Log One Reward Curve

A training dashboard should separate at least:

```text
raw task reward
optimized total reward
held-out success rate
reward mean, variance, and quantiles
each reward component
KL to reference and previous policy
token entropy and action entropy
trajectory length
tool-call and environment-error rate
all-zero and all-one group ratios
advantage mean, standard deviation, and sign by outcome
value loss and explained variance
gradient norm and PPO clip fraction
pass@1, pass@k, and trajectory diversity
monitor alert rate
```

This decomposition gives a practical diagnosis table:

| Observation | Likely failure |
|---|---|
| Task reward down, KL and clip fraction up | policy update too large |
| Reward down after entropy and diversity fall | exploration collapse |
| Raw reward normal, values and advantages abnormal | critic failure |
| Proxy reward up, held-out success down | reward hacking / overoptimization |
| All workers suddenly report zero | pipeline or environment incident |
| GRPO groups mostly all-zero or all-one | relative-advantage starvation |
| Total reward down, task success unchanged | KL, length, safety, or formatting component shift |

For long-running jobs, define automatic pause conditions before training begins. A crash investigation is much easier when the system preserves the checkpoint, optimizer state, rollouts, reward-code version, environment image, and critic version from both sides of the event.

---

## 10. A Practical Method-Selection Guide

| Situation | Good starting point | Why | Primary risk |
|---|---|---|---|
| Short trajectory, reliable outcome reward | GRPO / sequence-relative update | simple and critic-free | coarse credit, homogeneous groups |
| Repeated or restorable intermediate states | pairwise or GiGPO-style local comparison | matched-state relative credit | coverage and pair-selection bias |
| Large historical trajectory dataset | IQL plus advantage-weighted cloning | conservative offline reuse | critic error, dataset support |
| Sparse long-horizon environment | turn-level PPO plus value critic | temporal credit at action boundaries | value instability |
| Full trajectory reveals why a step failed | hindsight critic / HCAPO-style labeling | uses downstream evidence | narrative rather than causal credit |
| Environment can branch and rerun | counterfactual advantage | strongest local causal evidence | rollout cost and stochasticity |
| Hidden tests or backend truth exist | privileged critic | denser, better-informed training signal | leakage and shortcut learning |
| Judge is weak or gameable | held-out monitor plus verifier ensemble | reduces direct optimization pressure | higher evaluation cost |

This table is not a leaderboard. The right choice depends on what evidence the environment can produce.

---

## 11. The Training Stack I Would Build

### Stage 0: Make the environment auditable

Before RL, version:

- the task distribution;
- environment image and initial state;
- action grammar;
- verifier code;
- hidden tests;
- reward components;
- trajectory and tool-call schema.

Run corrupted-action negative controls. If deleting the test suite, refusing every task, or terminating immediately receives a good score, critic training is not the first problem.

### Stage 1: Create behavioral coverage

Use demonstrations, guided rollouts, recovery traces, and diverse failures. A sparse terminal reward cannot teach from states the actor never reaches. Keep explicit train/eval splits over task templates, environment states, and verifier variants.

### Stage 2: Establish the cheapest stable credit baseline

Start with outcome reward and group-relative or pairwise learning. This reveals whether a learned critic is necessary. Log group degeneracy and branch coverage.

### Stage 3: Add a turn-level critic

Train $V(s_t)$ or $Q(s_t,a_t)$ on complete trajectories. Where possible, let the critic see structured training-only evidence, but test each privileged feature with ablations. Calibrate turn predictions against future executable success, not only an LLM rubric.

### Stage 4: Add hindsight and counterfactual data selectively

Do not rerun every turn. Use the critic to identify high-uncertainty or high-impact branch points, then spend environment compute on those counterfactuals. Add the verified branches back into critic and policy training.

### Stage 5: Optimize conservatively

Use bounded policy updates, target-KL stopping, gradient clipping, and checkpoint rollback. Keep offline or replayed data close enough to the current policy that importance weights and critic targets remain meaningful.

### Stage 6: Hold out monitoring

Keep at least one monitor, verifier variant, or reward model out of the actor's direct training objective. Use it to detect reward hacking, distribution shift, and monitorability loss. Refresh adversarial evaluations as the policy improves.

### Stage 7: Stop on real performance, not proxy reward

The stopping metric should be held-out task success under a verifier the actor did not directly optimize, with constraint violations and side effects reported separately.

---

## 12. The Core Design Principle

Agentic RL needs richer critics because the trajectories are longer, the environment is partially observed, and the reward is delayed. But adding more learned reward does not automatically produce better credit.

The hierarchy of evidence should remain visible:

```text
verified environment effect
        > matched counterfactual execution
        > calibrated future-return prediction
        > privileged retrospective judgment
        > ungrounded textual plausibility
```

I would summarize the system design in one sentence:

> Reward behavior, verify outcomes, assign credit at turn boundaries, use hindsight to propose hypotheses, use counterfactuals to test them, and keep at least one monitor outside the actor's objective.

The critic is not the truth. It is an instrument for turning delayed evidence into useful updates. The system becomes robust when we can identify exactly what the critic knows, what it predicts, how the actor can game it, and which independent evidence can prove it wrong.

---

## Related Reading

- [From GRPO Outcome Rewards to Token-Level Advantage](/posts/2026-07-09-grpo-token-level-advantage/)
- [Scaling RL for White-Collar Work: The Environment Foundry](/posts/2026-07-03-scaling-rl-for-white-collar-work-environments/)
- [From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2604.09459)
- [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- [SWEET-RL](https://arxiv.org/abs/2503.15478)
- [Group-in-Group Policy Optimization](https://arxiv.org/abs/2505.10978)
- [Turn-PPO](https://arxiv.org/abs/2512.17008)
- [Hindsight Credit Assignment for Long-Horizon LLM Agents](https://arxiv.org/abs/2603.08754)
- [Agent-RLVR](https://arxiv.org/abs/2506.11425)
- [Detecting Misbehavior in Frontier Reasoning Models](https://openai.com/index/chain-of-thought-monitoring/)
- [Scaling Laws for Reward Model Overoptimization](https://openai.com/index/scaling-laws-for-reward-model-overoptimization/)

