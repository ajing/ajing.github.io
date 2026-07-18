---
author: Jing Lu
pubDatetime: 2026-07-18T17:30:00Z
title: "From Long CoT to Agent Swarms: The Documented Evolution of Kimi's Reinforcement Learning"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Reinforcement Learning
  - RLVR
  - Post Training
description: "A source-grounded history of Kimi's reinforcement-learning stack, from Kimi k1.5's long-context outcome RL and partial rollouts to K2's general RL and K2.5's multimodal GRMs and Parallel-Agent RL."
---

Most accounts of Kimi's reinforcement-learning work read like a model-release timeline: k1.5, K2, K2.5, and now K3. That framing misses the more consequential change.

Across the reports Moonshot has actually published, the **object being optimized keeps expanding**:

```text
one long completion
        ↓
an environment-grounded agent trajectory
        ↓
a multimodal trajectory producing code, research, or artifacts
        ↓
an orchestrator coordinating many agents
```

The algorithms changed along the way, but the deeper story is the widening boundary of the RL environment.

This post reconstructs that history from Moonshot's technical reports and official project pages. It separates three kinds of claims:

- **documented lineage**, where a report explicitly says one method builds on another;
- **parallel experiments**, which reuse related ideas but are not proven checkpoint ancestors;
- **unknowns**, where a release demonstrates capabilities without disclosing its RL recipe.

That distinction matters. Kimi-VL, Kimi-Dev, Kimina-Prover, and Kimi-Researcher reveal important parts of Moonshot's research program, but the public record does not establish a single checkpoint chain running through all of them.

## The Mainline We Can Actually Document

The public reports support one clean algorithmic lineage:

```text
Kimi k1.5
    │
    │ K2 explicitly adopts k1.5's policy-optimization foundation
    ▼
Kimi K2
    │
    │ K2.5 starts from K2 and modifies its RL objective
    ▼
Kimi K2.5
```

[Kimi K2](https://arxiv.org/abs/2507.20534) explicitly says its RL algorithm is based on the policy-optimization method introduced in k1.5. [Kimi K2.5](https://arxiv.org/abs/2602.02276) identifies K2 as its foundation and describes how its policy objective departs from the earlier formulation.

Everything else in this post should be read around that documented spine.

## Kimi k1.5: RL as Implicit Search

The January 2025 [Kimi k1.5 report](https://arxiv.org/abs/2501.12599) is the clearest starting point for Moonshot's published RL program.

Its central bet was that a language model did not need an explicit search tree, Monte Carlo tree search, a process reward model, or a learned value network to acquire planning behavior. If the model could generate a sufficiently long trajectory and receive a reliable terminal reward, it could represent search inside the autoregressive context itself:

```text
try a route
→ notice an inconsistency
→ backtrack
→ test another route
→ verify the answer
```

In that view, context length is not merely a capacity for reading longer documents. It is an axis of RL scaling because it increases the number of reasoning and recovery steps available inside one sampled trajectory.

### Training pipeline

The documented pipeline was:

```text
pretraining
→ vanilla supervised fine-tuning
→ small, high-quality long-CoT supervised fine-tuning
→ reinforcement learning
→ optional long-to-short transfer
```

The long-CoT warm start taught a basic reasoning grammar: planning, evaluation, reflection, and exploration. RL then allowed the model to discover its own trajectories rather than imitate a fixed reasoning trace.

### Outcome rewards rather than step labels

k1.5 trained on verifiable text and vision problems. Coding responses were executed against test cases. Mathematical answers were checked with rules or a learned correctness judge when equivalent free-form answers made exact matching unreliable.

The important choice was what the system did **not** reward directly. A bad intermediate step was not necessarily penalized if the model later detected the mistake, recovered, and reached a correct answer. The report argues that this can preserve useful trial-and-error behavior that a stepwise value function might suppress too early.

### Online mirror descent, not a casually renamed GRPO

k1.5 describes its method as a variant of online policy mirror descent. At each iteration, it samples several responses from a reference policy, centers their rewards using the empirical mean, applies relative-entropy regularization, updates the policy, and then uses the updated model as the next reference.

The use of a sampled mean reward may look similar to later group-relative methods, but the paper derives the update through a mirror-descent objective and supports off-policy data. It should not be flattened into “Kimi used GRPO” without qualification.

### Partial rollout: the systems idea that persisted

Long-context RL creates a severe long-tail problem. Some trajectories finish quickly; others consume the entire rollout budget. Waiting for the slowest sample wastes accelerators.

k1.5 introduced **partial rollout**:

```text
trajectory exceeds the current rollout budget
→ save the unfinished prefix in a replay buffer
→ continue it in a later RL iteration
→ reuse the old prefix instead of regenerating it
```

The overall training loop remained iterative and synchronous, while rollout workers operated asynchronously. The report also describes a colocated deployment in which training and inference engines take turns using the same accelerators, with a checkpoint engine moving updated weights between different sharding layouts.

### The first token-efficiency problem

Outcome RL made responses longer. Longer reasoning often improved accuracy, but it also increased training and inference cost.

k1.5 added a length reward that preferred shorter correct responses and penalized long incorrect ones. It also explored several long-to-short methods: selecting the shortest correct trajectory for supervised fine-tuning, constructing DPO pairs from short and long solutions, model merging, and a separate RL phase with a tighter rollout budget.

This accuracy-versus-compute tension did not disappear. K2.5 later returned to it with a more explicit alternating objective.

## Parallel Experiments: Expanding the Verifier

During the first half of 2025, several Moonshot projects tested how far outcome-based RL could travel. These projects are evidence of reusable ideas, not proof of one checkpoint lineage.

### Kimi-VL: the same reasoning framework crosses modalities

[Kimi-VL-Thinking](https://arxiv.org/abs/2504.07491) followed long-CoT SFT with an online mirror-descent RL stage similar to k1.5. Its prompt and reward space included pure-text and image-text reasoning. It added length rewards, curriculum sampling, and prioritized sampling.

The significance is narrow but real: long-context outcome RL was not restricted to a text-only decoder. The model could learn trajectories that combined visual perception, OCR, diagram interpretation, and symbolic reasoning.

### Kimina-Prover: a nearly ideal verifier

[Kimina-Prover](https://arxiv.org/abs/2504.11354) applied large-scale RL to Lean 4 proof generation. A proof either passes the formal verifier or it does not. The project generated whole proofs without training-time step feedback and, like k1.5, emphasized that strong results did not require MCTS, a value function, or a process reward model.

Formal verification offers unusually clean terminal supervision. It isolates a question that is harder in open-ended language tasks: what happens when the outcome judge is far more reliable than the policy?

### Kimi-Dev: rewards from real repositories

The June 2025 [Kimi-Dev release](https://moonshotai.github.io/Kimi-Dev/) moved from contest answers to actual repository edits.

Starting from Qwen2.5-72B, the project used roughly 150 billion tokens of GitHub issue and pull-request mid-training, a long-CoT cold start, and then k1.5-style policy optimization for code editing. Its reward was deliberately sparse and executable:

- a BugFixer patch received 1 only if it passed the complete ground-truth test suite;
- a TestWriter patch received 1 only if it reproduced the bug before the fix and passed after the fix;
- there were no format or process rewards in this RL stage.

The later [Kimi-Dev technical report](https://arxiv.org/abs/2509.23045) publishes unusually concrete settings: 1,024 training problems, 10 rollouts per problem, five training steps per RL iteration, a 64K maximum context, and a Kubernetes sandbox layer supporting more than 10,000 concurrent instances.

Kimi-Dev also exposed a practical curriculum pattern. Prompts with `pass@16 = 0` were initially excluded because they supplied no positive learning signal; as the policy improved, previously unsolved prompts were reintroduced.

## Kimi-Researcher: From Completion RL to Agent RL

The June 2025 [Kimi-Researcher report](https://moonshotai.github.io/Kimi-Researcher/) marks the largest conceptual transition.

k1.5 primarily optimized a sampled language completion. Kimi-Researcher optimized a trajectory whose next state depended on an external environment:

```text
state
→ thought and action
→ search, browser, or code tool
→ environment observation
→ updated state
→ next action
```

A trajectory could involve dozens of search queries and hundreds of thousands of context tokens. The model had to learn when to search, which evidence to inspect, how to reconcile conflicts, when to execute code, and when to stop.

### A distinct algorithmic disclosure

Kimi-Researcher says it was trained primarily with **REINFORCE** and stresses strict on-policy generation. During training, format-enforcing mechanisms in the inference engine were disabled so that tool-call behavior came from the model's own distribution.

Its published reward design included:

- format penalties for invalid tool calls or exceeding context and iteration limits;
- correctness rewards based on comparison with ground truth;
- a gamma-decay allocation on successful trajectories to favor shorter exploration paths;
- strategic removal of some negative samples to reduce the risk of entropy collapse.

This should not be silently substituted for K2's documented k1.5-based objective. Kimi-Researcher is a specialized agent system built on an undisclosed internal Kimi-series checkpoint.

### Rollout becomes an environment-scheduling problem

Kimi-Researcher described fully asynchronous actor rollouts, environmental interactions, and reward computation; a Gym-like interface; turn-level partial rollout; stateful tool sessions; and context management that retained important evidence while dropping unnecessary documents.

The bottleneck had moved. Generating tokens was still expensive, but waiting for search, browsers, code execution, and long-tail environments became equally important.

## Kimi K2: From RLVR to General Reinforcement Learning

The July 2025 [Kimi K2 report](https://arxiv.org/abs/2507.20534) pulled multiple capability domains into one post-training system and explicitly based its policy optimization on k1.5.

### Verifiable-reward Gym

K2 expanded RL across mathematics, STEM, logic, instruction following, faithfulness, coding, software engineering, and safety. Where possible, the reward came from a rule, unit test, execution environment, or task-specific verifier.

The agentic training pipeline also expanded the environment distribution. K2's report describes more than 3,000 real MCP tools, over 20,000 synthetic tools, thousands of generated agent configurations, rubric-based task generation, simulated stateful environments, and real execution sandboxes for coding.

### The self-critic bridge

Pure RL with verifiable rewards cannot directly optimize creative writing, helpfulness, or other open-ended qualities. K2 introduced a **Self-Critique Rubric Reward** for those domains.

The actor produced candidate responses. A K2 critic compared them using core rubrics, anti-reward-hacking rubrics, and task-specific human-authored rubrics. Most importantly, the critic was not treated as permanently fixed: on-policy rollouts from verifiable tasks were used to keep updating it.

The intended loop was:

```text
objective verifier signals from math, code, and environments
        ↓
continually refine the critic
        ↓
use the critic on open-ended tasks without direct verifiers
```

This was K2's move from RLVR toward what the report calls general reinforcement learning. It did not eliminate learned-judge risk, but it made the source of calibration explicit.

### Budget, forgetting, and exploration controls

K2 added several stabilizers around the inherited policy objective:

- task-dependent maximum token budgets with penalties for over-budget responses;
- an auxiliary pretraining loss on curated data to limit forgetting and narrow-task overfitting;
- a sampling-temperature schedule that favored exploration early and reliability later;
- Muon optimization during post-training;
- partial rollout for long-horizon agent tasks.

At the system level, K2 used colocated training and inference engines plus a distributed checkpoint engine. The report states that a complete parameter update for the one-trillion-parameter model took less than 30 seconds.

## K2 Thinking: A Capability Milestone, Not a Disclosed RL Recipe

The later [K2 Thinking release](https://www.kimi.com/blog/kimi-k2-thinking) demonstrated a model that could interleave reasoning with 200–300 sequential tool calls. It scaled test-time compute through both thinking tokens and tool-use steps, and its Heavy Mode rolled out eight trajectories in parallel before reflectively aggregating them.

The release also confirmed INT4 quantization-aware training for the MoE components during post-training.

It did **not** disclose the RL objective, reward composition, prompt distribution, or whether it reused K2's exact policy loss. Long tool-use evaluations are evidence of capability, not evidence of a particular training algorithm. Quantization-aware training is also a deployment-oriented post-training technique, not by itself an RL innovation.

## Kimi K2.5: Joint Multimodal RL and Parallel-Agent RL

The February 2026 [Kimi K2.5 report](https://arxiv.org/abs/2602.02276) is the next documented mainline step. K2.5 began from K2, added large-scale vision-text continual pretraining, and then performed supervised fine-tuning and reinforcement learning.

### Organizing RL by capability rather than modality

K2.5 trained text-only and multimodal queries jointly. Instead of defining separate “text RL” and “vision RL” organizations, it grouped training around capabilities such as knowledge, reasoning, coding, and agentic behavior.

Its verifiable rewards became correspondingly heterogeneous:

- rule-based correctness for reasoning and agent tasks;
- IoU- and F1-based grounding rewards;
- mask IoU for polygon segmentation;
- normalized edit distance for OCR;
- distance from ground truth for counting;
- LLM verification for synthesized visual puzzles.

### From self-critique to Generative Reward Models

K2.5 generalized K2's self-critique mechanism into **Generative Reward Models** applied across chat assistants, coding agents, search agents, multimodal traces, and artifact-generating agents.

These GRMs evaluated qualities including helpfulness, contextual relevance, instruction following, response readiness, level of detail, and artifact aesthetics. The report describes multiple alternative rubrics to reduce overfitting and reward hacking against one judge.

### Token-level clipping for rollout mismatch

K2.5 modified the earlier policy objective with token-level log-ratio clipping. Tokens whose training-policy and rollout-policy log probabilities diverged beyond a permitted interval had their policy gradients masked.

The report explicitly distinguishes this from standard PPO clipping: the decision is based on the log-ratio bound, not on the sign of the advantage. Moonshot describes the mechanism as important for stabilizing long-horizon, multi-step tool-use training where inference-training mismatch accumulates over many tokens.

### Toggle: efficiency without destroying test-time scaling

Rigid token budgets produced a new failure mode: a model trained to always reason briefly could lose the ability to use additional inference compute on genuinely hard problems.

K2.5 introduced **Toggle**, alternating between:

1. a budget-limited phase that rewarded solving eligible problems within a task-dependent token budget;
2. a standard scaling phase that allowed generation up to the full limit.

The budget constraint was activated only after the model's mean accuracy on a problem passed a threshold. This avoided demanding brevity before the policy had learned how to solve the task.

### PARL: the policy learns to organize other policies

K2.5's Parallel-Agent Reinforcement Learning changed the optimized decision maker again.

The trainable policy was an orchestrator. Subagents were frozen intermediate policy checkpoints. Their outputs were treated as environmental observations rather than differentiable actions. The separation reduced ambiguity about which component deserved credit for a final outcome.

In simplified notation, the orchestrator reward combined:

$$
R_{\text{PARL}}
= R_{\text{performance}}
+ \lambda_1 R_{\text{parallel exploration}}
+ \lambda_2 R_{\text{subtask completion}}.
$$

One auxiliary term discouraged **serial collapse**, where the orchestrator never delegated. The other discouraged **spurious parallelism**, where it created many useless agents merely to increase a concurrency statistic. Both auxiliary weights were annealed to zero so that final training converged on task quality.

K2.5 also introduced **critical steps** as a latency-oriented resource measure. A parallel group was charged according to its longest branch, analogous to the critical path in a computation graph. This rewarded decompositions that shortened end-to-end completion time rather than those that merely spawned more work.

The unified RL environment treated agent tasks as asynchronous coroutines, allowed recursive subtask rollouts, and reported support for up to 100,000 concurrent agent tasks.

## K2.6 and K3: Where the Public Evidence Stops

The April 2026 [K2.6 release](https://www.kimi.com/blog/kimi-k2-6) showed much longer coding sessions, proactive agents, and larger swarms. Its product system scaled to as many as 300 subagents and 4,000 coordinated steps in the examples described by Moonshot.

But the release did not publish a new RL algorithm, reward function, or training-data recipe. Those capability gains may combine pretraining, supervised data, RL, harness changes, context management, inference scaling, and product engineering. The evidence does not permit allocating the gains among them.

The same caution applies even more strongly to [Kimi K3](https://www.kimi.com/blog/kimi-k3). As of July 18, 2026, Moonshot has announced the 2.8-trillion-parameter architecture, native multimodality, a one-million-token context window, and quantization-aware training from the SFT stage onward. The launch post says that further training details will arrive with the technical report. It does not disclose K3's RL objective, rewards, rollout scale, or whether PARL is part of model training.

Any more specific K3 RL diagram published today would be reconstruction, not documentation.

## What Actually Evolved

The history is easiest to understand along four axes.

| Axis               | Early Kimi RL                         | K2                                                   | K2.5                                                                     |
| ------------------ | ------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| Optimized unit     | Long completion                       | Mixed completion and agent trajectory                | Multimodal trajectory and orchestrator episode                           |
| Reward             | Final correctness or execution        | RLVR plus self-critique rubric                       | Verifiers plus multi-rubric GRMs and PARL rewards                        |
| Efficiency problem | Long CoT and rollout tails            | Task budgets, forgetting, trillion-scale weight sync | Inference-training mismatch, length overfitting, parallel critical paths |
| Infrastructure     | Partial rollout and colocated engines | Large tool/task synthesis and real sandboxes         | Recursive asynchronous tasks and large-scale multi-agent rollout         |

The most durable Moonshot idea may not be one optimizer. It is the repeated attempt to preserve a simple terminal objective while expanding what counts as a trajectory and investing heavily in the systems required to generate those trajectories.

## Conclusion

The documented Kimi RL mainline is narrower than the model-release list but more interesting:

```text
k1.5
long-context outcome RL
+ online mirror descent
+ partial rollout

        ↓

K2
multi-domain RLVR
+ continually refined self-critic
+ budget, replay, and trillion-scale rollout infrastructure

        ↓

K2.5
joint text-vision RL
+ generative reward models
+ token-level mismatch clipping
+ Toggle
+ Parallel-Agent RL
```

k1.5 asked whether a model could learn search inside one long context. Kimi-Researcher showed what changed when that search touched external tools. K2 tried to extend reinforcement learning beyond tasks with perfect verifiers. K2.5 made the environment multimodal and made delegation itself a learned action.

K2.6 and K3 may extend that trajectory, but their public releases do not yet document how. Keeping that boundary visible is not a weakness in the story. It is what makes the story auditable.

## Primary Sources

- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
- [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491)
- [Kimina-Prover Preview](https://arxiv.org/abs/2504.11354)
- [Kimi-Dev official release](https://moonshotai.github.io/Kimi-Dev/) and [technical report](https://arxiv.org/abs/2509.23045)
- [Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- [Kimi K2 Thinking](https://www.kimi.com/blog/kimi-k2-thinking)
- [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)
- [Kimi K2.6: Advancing Open-Source Coding](https://www.kimi.com/blog/kimi-k2-6)
- [Kimi K3: Open Frontier Intelligence](https://www.kimi.com/blog/kimi-k3)
