---
author: Jing Lu
pubDatetime: 2026-08-14T19:35:03-07:00
title: "From Tool Calls to Policy Updates: A Reproducible Agent RL Stack"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Reinforcement Learning
  - Post Training
  - Systems
  - Evaluation
description: "A reproducible blueprint for Agent RL: environment snapshots, verifier contracts, credit assignment, three-policy semantics, asynchronous generation and training, partial rollouts, and runnable smoke tests."
---

> The hard part of Agent RL is not choosing PPO over GRPO. It is showing that a specific environment interaction produced a trustworthy gradient signal.

**Three contracts, one causal test, and a reference implementation that starts with a single-machine smoke run.**

Most introductions to Agent RL start with algorithm names: PPO, GRPO, RLOO, DAPO. That framing turns a systems problem into a survey of loss functions.

Suppose a trajectory earns reward 1. Before training on it, we still need to answer five questions:

1. Which environment state did the agent actually change?
2. Did an independent verifier assign the reward, or did the agent merely claim success?
3. Which turn, tool call, or token deserves credit?
4. Which `π_beh` sampled the actions, and how far is it from the `π_train` being updated?
5. When comparing checkpoints, did the prompt, tools, memory, sampler, and execution scaffold remain fixed?

If any question goes unanswered, the training curve may improve even though we still do not know what the model learned.

This article is not an Agent RL encyclopedia. Instead, it builds one minimal, runnable, and auditable loop:

```text
Task sampler → resettable environment → agent rollout
     ↑                                  ↓
held-out eval ← policy update ← credit ← verifier
```

The same reference implementation runs throughout the article. By the end, you should be able to plug in a terminal, browser, or API agent, run a small experiment, and only then decide whether you need more sophisticated credit assignment or distributed infrastructure.

The design rests on three contracts:

- **Transition Contract:** What happened in the environment?
- **Verification Contract:** Why does the terminal state deserve this reward?
- **Update Contract:** How does that reward map to action tokens and a policy update?

## 1. First define the training object: an agent is a POMDP policy

The key difference between an agent and single-turn RLVR is that its actions change the world it later observes.

A trajectory is:

```text
τ = (o₀, a₀, o₁, a₁, …, o_T)
```

The true state `s_t` is usually only partially observable. A database may contain values the agent has not read, a GUI may lag behind disk, and the user's intent may be only partially specified. The policy therefore conditions on history:

```text
πθ(a_t | h_t),  h_t = (o_≤t, a_<t)
```

This is naturally modeled as a [POMDP](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf). An LLM call can also act as a variable-duration macro-action, giving long-horizon agents an SMDP or options flavor.

### The first engineering choice: action granularity

At the environment layer, a tool-using agent usually emits one complete tool call as a single action:

```json
{ "kind": "tool", "name": "read_file", "args": { "path": "src/app.py" } }
```

At training time, however, the policy-gradient loss is computed over the tokens that encode that action:

```text
log πθ(a_t | h_t)
= Σ_{k ∈ action_mask_t} log πθ(y_{t,k} | h_t, y_{t,<k})
```

Tool results, system prompts, user text, and other agents' outputs are not actions taken by the current policy, so they must be masked out of the loss. Otherwise, the model learns to predict what the environment returned instead of what it should do next.

Safety should not be collapsed into a single scalar whose terms can cancel:

```text
maximize   E[R_task]
subject to E[C_unsafe] ≤ d
```

Severe authorization violations, grader tampering, and irreversible side effects should be handled with hard gates or constrained objectives. A high task reward must not be allowed to offset them.

### The policy boundary is larger than a checkpoint

For an agent, the policy being deployed and evaluated includes more than its weights:

```text
Π = (weights, system prompt, chat template, tool schemas,
     parser/canonicalizer, memory/compactor, retry/budget, sampler)
```

Even with frozen weights, changing the system prompt, history propagation, or tool-call template can shift results substantially. Crediting those gains to model capability would therefore confound training with the surrounding scaffold. A [systematic study of tool-calling evaluation pipelines](https://arxiv.org/abs/2606.00135) likewise finds sensitivity to seeds, system prompts, multi-turn templates, and history propagation.

At minimum, the implementation needs an immutable `PolicyManifest`:

```yaml
policy_version: ckpt-0007
weight_checksum: sha256:...
tokenizer_revision: sha256:...
system_prompt_hash: sha256:...
chat_template_hash: sha256:...
tool_schema_hash: sha256:...
parser_version: action-parser-v3
memory_policy: exact-history-v1
compactor_version: null
retry_budget: 1
step_budget: 8
sampler: { temperature: 1.0, top_p: 1.0, top_k: null }
```

Every rollout, update, and evaluation should point to this manifest, not just a checkpoint name. Use a paired sensitivity test: first hold the weights fixed and vary the scaffold to measure pipeline variance; then hold the scaffold fixed across checkpoints so that any remaining difference can be attributed to training.

---

## 2. Transition Contract: an environment is not a prompt wrapper

A training-ready environment needs at least four capabilities: exact reset, typed-action execution, authoritative-state export, and independent outcome verification. Local credit assignment also requires snapshot and restore.

```python
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class Action:
    kind: Literal["tool", "finish", "abstain"]
    name: str | None = None
    args: dict[str, Any] | None = None


@dataclass(frozen=True)
class Transition:
    observation: str
    done: bool
    state_version: str


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: str
    parent_snapshot_id: str | None
    env_version: str
    state_hash: str
    rng_state_hash: str


class AgentEnv(Protocol):
    def reset(self, task_id: str, seed: int) -> str: ...
    def step(self, action: Action) -> Transition: ...
    def export_state(self) -> bytes: ...
    def snapshot(self, label: str) -> SnapshotRef: ...
    def restore(self, snapshot: SnapshotRef) -> None: ...
    def fork(self, snapshot: SnapshotRef, branch_id: str) -> "AgentEnv": ...


class Verifier(Protocol):
    def score(self, task_id: str, final_state: bytes) -> dict[str, float]: ...
```

`Verifier` is deliberately separate from `AgentEnv`. The agent may observe the environment, but it must not control the scoring process. Provision fresh scorers, hidden tests, and reference artifacts under an independent identity in an immutable, read-only namespace outside the agent's write authority. Then verify the final artifact there. Do not trust the agent's own `pytest passed` message. A fresh rootless container is one layer of isolation, not a complete security boundary; high-risk arbitrary code execution still requires stronger isolation and least privilege.

### A snapshot is a state contract, not just a state dump

Even outcome-only RL requires every episode to reset reproducibly to its task-specific initial state. Precise replay, fault recovery, and turn-level branching make snapshots first-class environment objects. At minimum, distinguish three kinds:

| Snapshot                     | Created                                           | Purpose                                                                                                          |
| ---------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Base / reset snapshot**    | After task initialization, before the policy acts | Give sibling rollouts exactly the same starting point                                                            |
| **Branch snapshot**          | Immediately before an action is generated         | Fork continuations from the same `(h_t, s_t)` to estimate a local counterfactual advantage                       |
| **Sealed terminal snapshot** | After the episode ends                            | Preserve terminal state and side-effect evidence for an independent verifier; never expose it to the agent again |

A successful `restore` call is not enough. The real test of snapshot fidelity is:

```text
restore(S_t); replay(a_t, …, a_n)
    ⇒ identical observations, state hashes, and terminal reward

fork(S_t, branch_id=i) and fork(S_t, branch_id=j)
    ⇒ writes remain isolated and neither branch can mutate S_t
```

A valid snapshot must capture at least four layers:

1. **Mutable world state:** filesystem, database, browser cookies and local storage, application buffers, tool caches, and external memory written by the agent.
2. **Entropy sources:** environment RNG, mock clock, queue ordering, and controllable service responses. When the environment depends on the live internet or shared SaaS, claim only partial replay unless it uses simulation or record/replay.
3. **Provenance:** task, parent snapshot, environment image, schema/version, and content hash.
4. **Agent prefix:** a world snapshot does not contain `h_t`. Reconstructing a branch also requires the exact context tokens, retrieval and memory state, behavior-policy version, and sampling configuration.

The true branch point is therefore more than a bare `snapshot_id`:

```text
B_t = (snapshot_ref, context_token_ids, memory_ref,
       behavior_policy_version, sampling_config)
```

The snapshot backend depends on the environment:

| Environment             | Minimal snapshot backend                                                                                       | Common gap                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Python state machine    | Canonical JSON/MessagePack plus a content hash; copy-on-write branches                                         | Hidden global RNG or mutable singletons                             |
| Stateful API / database | An isolated database or schema per episode, plus transaction/savepoint or logical dump                         | External jobs, caches, or shared accounts do not roll back          |
| Terminal / container    | Immutable base image plus an isolated writable overlay; quiesce and take an application-level dump when needed | A filesystem layer omits RAM, sockets, and unflushed buffers        |
| Browser / GUI           | Server database plus browser profile, cookies/local storage, and a fixed clock                                 | A screenshot or DOM is not authoritative state                      |
| OS / resident process   | VM or microVM memory snapshot, otherwise an application-level checkpoint                                       | Restore is expensive; devices and external networks can still drift |

[OSWorld](https://arxiv.org/abs/2404.07972) provides initial-state setup and execution-based evaluators. [AppWorld](https://arxiv.org/abs/2407.18901) uses state-based tests for both task goals and collateral damage. Both show why “the page looks the same” is not a valid state-equivalence test. [BPO](https://arxiv.org/abs/2607.14171) uses intermediate snapshots and sibling forks for local credit, but remains a 2026 preprint.

### Components you can use directly

| Layer                       | Minimal implementation                                      | When you need more                                                             |
| --------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Terminal / code environment | Python adapter + Docker + `pytest` or state diff            | Terminal-Bench-style container tasks                                           |
| Stateful API                | In-memory database + typed tools                            | [τ-bench](https://arxiv.org/abs/2406.12045)- or ToolSandbox-style environments |
| Web / GUI                   | Version-pinned local website + browser driver               | WebArena- or OSWorld-style evaluator                                           |
| Snapshot substrate          | Canonical state + content-addressed manifest                | Per-episode databases, copy-on-write overlays, or VM snapshots                 |
| Evaluation harness          | Independent identity + read-only grader namespace           | [UK AISI Inspect](https://inspect.aisi.org.uk/)                                |
| High-risk execution         | Rootless container, no host secrets, network off by default | gVisor or microVM, plus episode-scoped credentials                             |

One of the most neglected environment properties is **state authority**. Disk, an application buffer, and the rendered GUI can disagree. “It looks correct on screen” does not prove that the persistent state is correct. Bind every mutation and completion claim to a state version, and fail closed when that version has drifted.

---

## 3. Transition Contract: a rollout must be sufficient to reconstruct the gradient

A prompt-and-response log is not enough. A training trace should look more like this JSONL record:

```json
{
  "run_id": "run-0042",
  "task_id": "json-repair-017",
  "seed": 11,
  "env_version": "sha256:...",
  "base_snapshot_id": "snap-json-repair-017-v3",
  "behavior_policy_version": "ckpt-0007",
  "policy_manifest_hash": "sha256:...",
  "behavior_weight_checksum": "sha256:...",
  "generator_build": "vllm:...",
  "rollout_started_at": "...",
  "rollout_finished_at": "...",
  "parent_run_id": null,
  "branch_point_turn": null,
  "turn": 3,
  "snapshot_before": "snap-run-0042-t3",
  "state_before": "sha256:...",
  "context_segment_id": 0,
  "memory_state_ref": "sha256:...",
  "compactor_version": null,
  "prompt_token_ids": [101, 202],
  "action_token_ids": [303, 404],
  "sampler_config": { "temperature": 1.0, "top_p": 1.0, "top_k": null },
  "behavior_logprobs": [-0.21, -0.08],
  "raw_model_logprobs": [-0.21, -0.08],
  "action_mask": [1, 1],
  "tool_name": "write_file",
  "tool_args": { "path": "answer.json" },
  "observation": "write succeeded",
  "state_after": "sha256:...",
  "reward_components": null,
  "termination_reason": null,
  "continuation_parent": null
}
```

Five rules are non-negotiable:

1. Store the **exact token IDs and behavior log probabilities of the distribution that actually sampled them**. If temperature, top-p, or top-k transformed the distribution, also retain raw-model log probabilities when possible. Retokenizing later can change the sequence because the chat template, BPE revision, or tool-call rendering changed.
2. Pin one `behavior_policy_version` for the entire trajectory. Do not hot-swap weights halfway through an episode in the first implementation.
3. Record infrastructure errors, timeouts, and environment startup failures separately. Never turn them silently into reward 0.
4. Preserve the raw observation or an immutable object reference. A model-generated summary is not sufficient for replay.
5. For a branch rollout, record its parent run, branch point, and snapshot manifest. The same snapshot without the same agent prefix is not the same decision state.

There are two distinct reproducibility claims. Given recorded actions, **environment replay** should recover the same observations, state hashes, and reward. By contrast, re-running GPU sampling need not reproduce every token across kernels and inference engines, so token-identical sampling should not be a cross-platform requirement. An invalid action should consume a declared step, return a structured error, and produce no undeclared side effects; the harness must not silently repair it for the model.

Start synchronously: freeze a checkpoint, finish a complete rollout batch, and then perform exactly one optimizer step. Introduce asynchrony only after this loop is correct. Otherwise, asynchrony can make trajectories stale and conceal off-policy bias in the gap between `π_beh` and the current `πθ`.

Hugging Face generation is sufficient at small scale. For throughput, use [vLLM](https://arxiv.org/abs/2309.06180) or [SGLang](https://arxiv.org/abs/2312.07104). In either case, verify that the server returns the true sampled token IDs and log probabilities rather than text that is retokenized after generation.

---

## 4. Verification Contract: verifier quality sets the reward ceiling

The most dangerous bug in Agent RL is assigning a high reward to an incorrect artifact. That error does more than corrupt one evaluation example: it contributes a gradient in the wrong direction.

Layer verifiers in this order:

```text
authoritative state check / hidden tests / compiler
                         ↓ hard gate
rules and policy invariants
                         ↓
independent semantic judge for residual open-ended semantics
                         ↓
human audit of high-risk samples
```

A useful verifier should return more than `reward: 1`:

```python
score = {
    "task_success": 1.0,
    "policy_violation": 0.0,
    "collateral_damage": 0.0,
    "tool_calls": 6.0,
    "tamper_detected": 0.0,
}

if score["tamper_detected"]:
    raise InvalidMeasurement("quarantine this episode")

reward_task = score["task_success"]
constraint_cost = (
    score["policy_violation"]
    + score["collateral_damage"]
)
```

Preserve the reward vector before deciding which terms enter the objective. Tampering invalidates the measurement itself, so quarantine the sample. Put policy violations and collateral damage behind hard constraints or in a separate constraint cost. You can then plot a safety–utility frontier instead of guessing why a composite reward moved.

On a representative audit set adjudicated by humans or a stronger external program, estimate at least two kinds of verifier error:

- **False positive:** an incorrect result is accepted and can reinforce incorrect behavior in the policy.
- **False negative:** a correct result is rejected, wasting useful experience and potentially making the policy overly conservative.

A low false-positive rate comes with trade-offs: a conservative hybrid verifier may achieve it by rejecting more correct outputs. Report FP, FN, denominators, and confidence intervals together. Report ECE or Brier score only when the verifier emits interpretable probabilities, and report disagreement only when multiple verifiers exist. Do not force probability calibration onto a deterministic compiler or hidden test. In its studied settings, [reward-model overoptimization](https://arxiv.org/abs/2210.10760) also shows that optimizing a proxy can progressively separate proxy reward from true quality. That result is not a direct estimate of Agent-verifier error rates.

---

## 5. Update Contract: removing a critic does not solve long-horizon attribution

Suppose we sample `G` sibling trajectories from the same task, initial state, and behavior `PolicyManifest`. The simplest critic-free estimator is RLOO:

```text
A_i = R_i - (1 / (G - 1)) Σ_{j ≠ i} R_j
```

[RLOO](https://aclanthology.org/2024.acl-long.662/) uses the mean reward of the other samples as the baseline for trajectory `i`. GRPO commonly normalizes within a group:

```text
A_i = (R_i - mean(R₁:G)) / (std(R₁:G) + ε)
```

Both methods avoid a separate critic. They are well suited to settings with a strong outcome verifier, relatively short episodes, and multiple samples from the same task. [DeepSeekMath](https://arxiv.org/abs/2402.03300) is the primary public source for GRPO.

Every trajectory in a group must share the same task, initial environment seed, and behavior `PolicyManifest`. Normalizing across unrelated task difficulties would conflate relative task difficulty with action quality.

If siblings are conditionally independent given the task and checkpoint, and each trajectory succeeds with probability `p`, then a Bernoulli-reward group has no relative signal with probability:

```text
P(zero variance) = p^G + (1-p)^G
```

Easy tasks produce all successes; tasks that are too hard produce all failures. Neither case supplies a group-relative gradient, so task sampling becomes part of the RL estimator. But **reward variance means only that a batch can produce a gradient; it does not mean the experience will improve held-out capability.**

### Optimization unit: token, turn, and trajectory are different objectives

If the estimand is expected episode return, the score-function term is `A_i Σ_t log π(a_i,t|h_i,t)`. Dividing it by a quantity that depends on the sampled trajectory length changes that estimand. The following objectives may appear to differ only in where the `mean` is taken, but they weight the length distribution differently:

```text
L_episode  = (1/B) Σ_i Σ_t ℓ_i,t
L_len_norm = (1/B) Σ_i [(1/T_i) Σ_t ℓ_i,t]
L_token    = (1/Σ_i T_i) Σ_i Σ_t ℓ_i,t
```

`L_episode` is the episode-return policy-gradient objective. `L_len_norm` gives each trajectory approximately equal total scalar weight, so each token in a long trajectory receives a smaller coefficient. `L_token` forms a ratio estimator using the batch's realized token count. Splitting a multi-turn trajectory into turn samples and averaging changes the episode weights yet again. The latter two may be deliberate variance- or length-control surrogates, but neither is the original expected-return estimand.

Dividing by `std(R_group)` is not merely a numerical stabilization trick. It reweights task groups by their within-group reward scale and becomes especially sensitive near zero variance. [Dr.GRPO](https://arxiv.org/abs/2503.20783) modifies the objective in response to group-standard-deviation and response-length normalization bias. [STEPO](https://arxiv.org/abs/2607.09773) explicitly constrains turn-level credit mass in multi-turn settings. STEPO is a 2026 method and belongs in an ablation, not in the default recipe.

The minimal `LossReducer` comparison is:

```yaml
reduction: episode_sum # vs length_normalized vs token_mean vs turn_mass_conserving
group_reward_scale: none # vs std
```

Alongside final success, report per-episode gradient mass by success/failure, trajectory length, turn count, and task family. Otherwise, reward may rise simply because the optimizer learned to favor one length band.

### A dense score is not causal credit

Broadcasting one `A_i` to every action token tells us only that the trajectory outperformed its siblings. It does not identify which tool call caused the success. An important early exploratory action and a late, irrelevant verbose action receive the same sign. Adding a score at every step does not automatically solve the problem either: a step score can correlate with success even when the action does not change the probability of success.

### When the credit estimator needs an upgrade

| Observed problem                            | Better technique                                                       | Cost or assumption                                          |
| ------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| Short task with a strong outcome verifier   | RLOO / GRPO                                                            | Coarse trajectory-level credit                              |
| Trustworthy step reward exists              | PPO + critic + [GAE](https://arxiv.org/abs/1506.02438)                 | Critic cost and bias                                        |
| Only redistribute total credit across turns | STEPO-style mass redistribution                                        | Reweights turns; does not identify counterfactual causality |
| The same state recurs within a trajectory   | [GiGPO](https://arxiv.org/abs/2505.10978)-style anchor-state grouping  | Requires reliable state matching                            |
| Frozen reference and known target answer    | [TRACE](https://arxiv.org/abs/2607.13988)-style tool-boundary TD proxy | Depends on reference quality; 2026 evidence                 |
| Environment supports snapshot and restore   | [BPO](https://arxiv.org/abs/2607.14171)-style sibling branches         | Fork cost and restore fidelity; 2026 evidence               |
| Trajectory crosses the context window       | Segment critic or compaction with cross-segment credit                 | A summary can lose sufficient state                         |

In a snapshotable environment, fork `K` continuations from the same state `s_t`, each beginning with a sampled action, and roll them to termination:

```text
S_t = env.snapshot();  H_t = exact_agent_prefix()

for k in 1..K:
    env_k = env.fork(S_t, branch_id=str(k))   # same env RNG state
    τ_t,k = rollout(env_k, prefix=H_t, policy=π_beh,
                    policy_sampling_seed=k)

A_t,k = G_t,k - (1 / (K - 1)) Σ_{j ≠ k} G_t,j
```

Because sibling continuations share both world state and agent prefix, the comparison is closer to a local counterfactual than subtracting rewards from unrelated initial trajectories. `branch_id` isolates writes; it must not silently alter environment randomness. Exploration differences should come only from policy sampling. Every branch also needs the same remaining step and token budgets, plus an isolation test. Otherwise, a higher return may merely reflect more compute. This argument relies on faithful snapshots and controlled environment randomness; it is not universal causal identification for open environments.

Whatever `CreditEngine` you choose, add a small causal audit. At a fixed snapshot, delete, replace, or resample actions assigned high credit. Then measure the change in terminal return and report the rank correlation between estimated credit and intervention delta. This cannot prove global optimality, but it can reject a dense score that is unrelated to action effect. Keep an outcome-only arm so that the process reward does not become a new attackable proxy.

DAPO and DPPO primarily change sampling, aggregation, or trust-region control; neither identifies which turn caused success. [DAPO](https://arxiv.org/abs/2503.14476) uses dynamic sampling to reduce zero-variance groups. [DPPO](https://arxiv.org/abs/2602.04879), a 2026 preprint, constrains updates using more direct KL/TV divergence estimates. Credit assignment remains a separate design decision.

---

## 6. Update Contract: separate the estimator, objective, and system

### Name the three policies

A reliable implementation cannot collapse every previous model into `old_policy`:

| Symbol   | Role                                                       | What must be stored                                                             |
| -------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `π_beh`  | The behavior distribution that actually sampled the action | Generator build, complete sampler/grammar, and per-action-token log probability |
| `π_prox` | Center of clipping or the trust region                     | Checkpoint version and whether it equals `π_beh`                                |
| `π_ref`  | Frozen capability anchor                                   | Reference revision, KL direction, and estimation method                         |

In synchronous one-step training, `π_beh = π_prox` is common. Under asynchrony, replay, or generator/trainer numerical mismatch, do not assume that equality. If behavior data are corrected toward a proximal policy and the update is then constrained around that proximal policy, the conceptual decomposition is:

```text
w_off  = π_prox(a|h) / π_beh(a|h)
r_prox = π_θ(a|h)    / π_prox(a|h)

π_θ(a|h) / π_beh(a|h) = r_prox · w_off
```

An estimator may truncate, reject, or combine these ratios, but it must say which quantity it changes. [PPO](https://arxiv.org/abs/1707.06347) provides a proximal update only when its data and old-policy semantics hold. [AReaL](https://arxiv.org/abs/2505.24298) explicitly separates asynchronous behavior and proximal policies.

`π_ref` has a different role: it is not the importance-sampling denominator. `log πθ - log πref` estimates a target KL only under a stated sampling distribution, and averaging it over stale behavior samples does not automatically recover the current policy's forward KL. Specify the KL direction, the sampling distribution, whether you use a full-vocabulary audit or a non-negative estimator, and whether KL enters the reward or the loss—not both by accident.

### Align the granularity of reward, ratios, and clipping

Agents often receive only a sequence-level outcome reward. Distinguish a true change-of-measure ratio from a length-normalized stability score:

```text
token IS ratio:       r_t = exp(log πθ(a_t|h_t) - log πbeh(a_t|h_t))
trajectory IS ratio:  R_IS = exp[Σ_t log r_t]

turn normalized score:     s_turn = exp[(1/T_turn) Σ_t log r_t]
episode normalized score:  s_seq  = exp[(1/T_episode) Σ_t log r_t]
```

`R_IS` is the actual trajectory importance ratio, but its variance grows explosively with horizon. `s_turn` and `s_seq` are geometric-mean surrogates that deliberately change the target; they do not provide unbiased trajectory-level off-policy correction. [DAPO](https://arxiv.org/abs/2503.14476) uses a token-level policy loss with asymmetric clipping. [GSPO](https://arxiv.org/abs/2507.18071) instead uses a length-normalized sequence likelihood-ratio score with sequence-level clipping, with particular emphasis on MoE stability. These choices are not interchangeable implementation details. For a multi-turn agent, include a turn-normalized score as a third ablation and report token, whole-turn, and whole-episode clip fractions separately. Report a rejection fraction only if the system actually discards samples.

The simplest baseline applies under a strict synchronous contract. Every rollout must come from the current complete `PolicyManifest`; `π_beh = π_prox = πθ` at the start of gradient computation; generator and trainer log probabilities must pass the consistency gate; and the fresh batch must receive exactly one optimizer step. Under these conditions, the simplest objective for expected episode return is RLOO/REINFORCE plus KL:

```text
L_pg = -(1/B) Σ_i A_i · [Σ_{k∈M_i} log πθ(y_i,k | y_i,<k)]

L = L_pg + β KL(πθ || πref)
```

This baseline has no critic and no replay. At that first step, there is no behavior mismatch for ratio clipping to correct. Dividing by the sample-specific `|M_i|` would instead switch to the length-normalized surrogate from the previous section.

Whenever `πθ ≠ π_beh`—because of multiple epochs over a batch, stale async or replay data, or generator/trainer mismatch that requires correction—the implementation must explicitly choose importance correction, clipping, rejection, or quarantine. The standard PPO-style surrogate below covers only the special case `π_prox = π_beh`. When all three policies differ, return to the three-policy contract:

```text
ρ_t(θ) = exp(log πθ(a_t|h_t) - log πbeh(a_t|h_t))

L = -E[min(ρ_t A_t,
           clip(ρ_t, 1-ε, 1+ε) A_t)] + β KL(πθ || πref)
```

[PPO](https://arxiv.org/abs/1707.06347) clips the surrogate likelihood ratio on sampled actions; it does not directly constrain the full policy change. Monitor `clip_fraction`, KL, entropy, gradient norm, and importance-ratio tails together. The preregistered `LossReducer` must control the reduction; framework defaults must not silently decide whether tokens, turns, or trajectories receive equal weight.

The minimum training loop is deliberately simple:

```python
for task_batch in sampler:
    behavior = registry.freeze_current()

    groups = collector.rollout(
        policy=behavior,
        tasks=task_batch,
        group_size=8,
    )

    verified = verifier.score_in_clean_room(groups)
    batch = validator.require_fresh_finite_complete(groups, verified)
    advantages = rloo(batch.episode_rewards)

    loss = reinforce_with_kl_loss(
        new_logprobs=model.logprobs(batch),
        reference_logprobs=reference.logprobs(batch),
        advantages=advantages,
        mask=batch.action_mask,
        kl_beta=0.02,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    registry.publish_new_checkpoint()
```

This structure keeps three decisions separate:

- `advantages` selects the credit estimator.
- `reinforce_with_kl_loss` selects the update objective. Whenever `πθ ≠ π_beh` because of batch reuse, async or replay staleness, or a generator–trainer mismatch requiring correction, use an objective with explicit ratios or correction.
- `collector/registry` determines whether the admitted data are near-on-policy.

As a result, you can swap RLOO for GAE or branching credit without rewriting the environment or verifier.

### Average entropy can hide exploration collapse

Recent work on mathematical-reasoning RLVR studies gradient covariance and high-entropy minority tokens, finding that a small subset of tokens can carry a disproportionate share of the exploration signal. [Entropy Mechanism](https://arxiv.org/abs/2505.22617) and [High-Entropy Minority Tokens](https://arxiv.org/abs/2506.01939) motivate a testable hypothesis for agents: do those tokens correspond to tool choices or behavioral forks? The reasoning evidence does not establish that mapping.

An `ExplorationMonitor` should stratify entropy and gradient mass by advantage sign, entropy quantile, turn index, tool/action type, and success/failure. Test whether high-entropy tokens align with actual branch or action interventions, and plot held-out `pass@k` and tool-sequence diversity alongside `pass@1`. These views help distinguish justified convergence under reward, distributional bias from clipping, and premature collapse at consequential decisions.

### Asynchrony between generator and trainer changes the data distribution

First separate two roles that are often both called the actor:

- A **generator / rollout worker** holds an inference-only behavior policy `π_beh^v`, interacts with the environment, and writes immutable trajectories.
- A **trainer / learner** holds the master parameters, optimizer state, and current training version `π_train^u`; it consumes trajectories and publishes new weights.

The data flow is:

```text
Task queue → Generator pool (π_beh^v) → Environment
                    ↓                     ↓
             exact tokens/logprobs → immutable trajectory queue
                                              ↓
Verifier → complete group → Trainer (π_train^u) → staged checkpoint
                                                  ↓
                         atomic registry ← weight sync ← checksum/ACK
                              ↓
                    Generator pool (π_beh^(v+1))
```

Track both system lag and sampled-action drift:

```text
Δ_version = u - v
Δ_time    = trainer_consume_time - rollout_finish_time
D_sample  = (1/|M|) Σ_{t∈M(τ~π_beh^v)}
            |log π_train^u(a_t|h_t) - log π_beh^v(a_t|h_t)|
```

`Δ_version` is system age. `D_sample` is only a drift diagnostic over action tokens sampled from `π_beh^v`, restricted to the action mask `M`; it is not a policy-space distance. A one-version lag can correspond to very different amounts of drift under different learning rates, KL penalties, update sizes, and sampling distributions. Enforce both a version gate and a ratio or drift gate.

The synchronous/asynchronous distinction is not binary. It spans at least three data semantics:

| Mode                          | Generator–trainer relationship                                              | Data semantics                                                                    |
| ----------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Synchronous barrier**       | `π^v` finishes every group before the trainer publishes `π^(v+1)`           | `Δ_version=0`; cleanest, but blocked by the longest trajectory                    |
| **Pipelined / bounded async** | Generators continue; trainer admits only complete groups with `Δ_version≤L` | Bounded staleness; call it near-on-policy only if drift and ratio gates also pass |
| **Unbounded async / replay**  | Trainer can repeatedly consume arbitrarily old data                         | Genuinely off-policy; an unchanged on-policy recipe is invalid                    |

Set `L=0` in the reference implementation, and add `L=1` only as an explicit ablation. Do not begin with “train whatever is in the queue.” For RLOO or GRPO, every sibling in a group must also come from the same behavior version. Mixing `π^v` and `π^(v+1)` makes group differences reflect both action quality and policy drift.

Schedule rollouts by a complete-group lease, not by whichever trajectory arrives first:

```text
lease = (group_id, behavior_manifest_hash,
         base_snapshot_id, sampler_config_hash)
```

This group lease binds every sibling to the same behavior manifest, base snapshot, and sampler until the full group has completed verification. Only then should the trainer compute their advantages. If one sibling times out, discard the group or apply a preregistered fallback estimator. Never fill the missing slot later with a sample from a newer policy.

#### What happens to an in-flight episode?

A long agent rollout may still be running when the trainer publishes a new version. Only three choices have clear semantics:

1. **Finish old:** let the old generator finish the episode and route only new tasks to the new version. This has the cleanest semantics but requires double buffering or temporary retention of old workers.
2. **Abort and restart:** discard the partial trajectory and restart from the base snapshot under the new version. This is statistically clean but wastes rollout compute.
3. **Mixed-policy continuation:** switch weights mid-episode. A strict on-policy baseline should forbid this mode. If you use it, store the behavior version and log probability for every action or token and choose an estimator designed for mixed-policy data.

`pause → load new weights → resume` is a serving feature, not a proof of RL correctness. If a single in-flight generation spans two parameter versions, quarantine it and regenerate it. If an episode switches versions between turns, label it as mixed-policy. A KV cache created under the old weights is invalid under the new weights. [vLLM's native RL APIs](https://vllm.ai/blog/2026-05-28-native-rl-apis) support weight transfer and `abort`, `wait`, and `keep` pause modes, but the algorithm still has to choose one of the semantics above.

[DORA](https://arxiv.org/abs/2604.26256), a 2026 preprint, takes a different approach: it keeps multiple rollout-policy versions alive so that long trajectories can finish on their original version, then applies bounded-staleness admission at the trainer. It is a useful reference for multi-version serving, not yet established practice.

#### Partial rollouts, truncation, and compaction change the data semantics

An incomplete episode can represent several statistically different events:

```text
agent_finish | env_terminal_success | env_terminal_failure
task_budget_exhausted_terminal
max_step_truncation | max_token_truncation
tool_timeout | infra_abort | scheduler_cancel
```

The first three are policy or environment terminal events, although the verifier still decides success. Exhausting a finite-horizon budget under a preregistered task contract can also be a legitimate terminal failure. By contrast, ad hoc harness truncation, tool timeout, infrastructure abort, and scheduler cancellation are censored or system events. What matters is not the enum name, but whether the budget was part of the task definition in advance. Silently assigning reward 0 to censored or system events turns censoring into failure and systematically reweights long tasks.

[APRIL](https://arxiv.org/abs/2509.18521) overprovisions requests, admits a target number of completed samples, and carries unfinished prefixes into later iterations to reduce rollout tail latency. [RollPacker](https://arxiv.org/abs/2509.21009) provides a useful comparison because it tries to preserve synchronous semantics. APRIL's first-completion admission rule **may** induce a temporal curriculum that favors fast or short trajectories. This is an inference to test by length, difficulty, and reward—not a result established by the paper.

To continue a partial rollout, the `TrajectorySegmentStore` must contain:

```text
segment_id, continuation_parent,
snapshot_id, exact_context_token_ids, memory_state_ref,
behavior_policy_version, behavior_logprobs, sampler_config,
remaining_step/token_budget, termination_reason
```

Preregister four comparisons: continuation pinned to the old version, mixed-version continuation, abort and restart, and synchronous completion. A continued RLOO or GRPO group must preserve its group semantics. Do not replace a missing sibling with one generated later by a new policy.

Context compaction is part of the policy boundary, not a lossless form of log compression. The summary determines what the policy sees next and may omit state needed by the verifier. [CompactionRL](https://arxiv.org/abs/2607.05378) jointly optimizes summary generation and cross-segment credit, providing recent 2026 evidence. At minimum, record pre- and post-compaction token hashes, the compactor model and version, summary token IDs, and the snapshot at which compaction occurred. Keep a no-compaction control, and test summary omission separately from restore fidelity.

#### Weight publication must be atomic

Never accept a new request while only some tensors have been loaded or only some TP/PP/EP ranks have received the update. Use at least the following publication protocol:

```text
trainer completes u+1
→ write immutable manifest (model/tokenizer/template/checksum)
→ stage weights into a shadow slot on every inference rank
→ every rank verifies checksum and ACKs
→ atomically switch active_version at the declared policy-lease boundary
→ if a logical episode continues, invalidate KV cache and prefill under new weights
→ only then route subsequent requests to u+1
```

Do not confuse the group lease above with the episode-level policy lease. In the pinned-policy baseline, each episode-level lease prevents mid-episode weight switching and ends at that episode's boundary. Only an explicit mixed-policy experiment may switch atomically at a recorded segment or turn boundary, with the continuation labeled under a new behavior version. Never hot-swap silently at an arbitrary token.

Colocated systems can time-share devices and reshard between training and inference. Separate GPU pools require NCCL, CUDA IPC, RDMA, or checkpoint/object-store transport. [HybridFlow/verl](https://arxiv.org/abs/2409.19256) separates RL dataflow from model placement and resharding. LoRA reduces the number of bytes transferred, but `base_revision + adapter_revision` still forms one indivisible policy version.

The trainer might use BF16 while the generator uses FP8 or another quantized format. Even with identical version IDs and checksums, differences in kernels, parallel layouts, MoE routing, constrained decoding, and sampler implementations can produce different log probabilities. This **training–inference mismatch** is distinct from stale data:

```text
δ_TIM,t   = log π_generator^v(a_t|h_t) - log π_trainer^v(a_t|h_t)
δ_stale,t = log π_trainer^v(a_t|h_t)   - log π_trainer^u(a_t|h_t), u > v
δ_total,t = log π_generator^v(a_t|h_t) - log π_trainer^u(a_t|h_t)

δ_total,t = δ_TIM,t + δ_stale,t
```

Measure pure TIM by comparing the generator and trainer at the same version `v`. Measure pure staleness by loading versions `v` and `u` in the same trainer backend. Comparing generator `v` with trainer `u` measures only the total gap. Retain the signed per-token deltas; report their signed mean together with the p95 and maximum of `|δ|`. Subtracting absolute values does not yield an error decomposition.

At a frozen checkpoint, a `LogprobConsistencyAuditor` should replay exact prefixes and sampled token IDs through both backends. It should report TIM metrics, the sequence log-perplexity difference, and a non-negative `k3` estimator. Preregister tolerances on the actual hardware and include a pure BF16 same-engine baseline. [Diagnosing Training–Inference Mismatch](https://arxiv.org/abs/2605.14220) reports that even small numerical differences can disrupt training after policy drift is isolated. [verl's rollout-correction implementation](https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md) exposes related metrics, importance correction, and rejection hooks. These are implementation references; correction does not replace a zero-mismatch diagnostic.

The tokenizer, chat template, tool rendering, grammar, quantization, routing replay, and inference engine all belong in `PolicyManifest`. Without them, apparent ratio drift may come from an implementation mismatch rather than a policy update.

#### What should happen to stale data?

PPO clipping limits an update on sampled actions, but it cannot turn severely stale data back into on-policy data. Meanwhile, multiplying importance ratios across a long trajectory creates extreme variance.

Behavior log probabilities must describe the **actual sampler distribution**. If the generator applies temperature, top-p, or top-k but stores only raw-model log probabilities, the importance-ratio denominator is wrong. Truncation may also violate the support condition required for off-policy correction. The cleanest baseline uses temperature 1 without top-k or top-p truncation. If truncation is retained, record the fully transformed sampling probability while noting that logging it does not restore missing support.

Use this order of operations:

1. First bound how far off-policy the data may be, using a version-lag cutoff and groups that are homogeneous in behavior version.
2. Store rollout-time behavior log probabilities; recompute current probabilities in the trainer; monitor ratio p50/p95/p99 and effective sample size.
3. For mild drift, use ratio clipping, a KL gate, or explicit sample rejection.
4. Consider truncated importance correction such as [IMPALA/V-trace](https://arxiv.org/abs/1802.01561) only when a turn-level critic already exists.
5. Quarantine trajectories beyond the preregistered lag or ratio gate, or retain them for offline analysis; never train on them silently.

An asynchronous stack is also a queueing system. If the generator rate `λ_gen` persistently exceeds the trainer admission rate `λ_train`, the queue grows and samples inevitably become staler. Monitor queue age and high-water marks, then apply generator backpressure, dynamic worker allocation, or stale-sample quarantine. Stratify stale and dropped rates by task family, trajectory length, and reward. Otherwise, the system may preferentially discard long, difficult episodes and silently alter the curriculum.

[AReaL](https://arxiv.org/abs/2505.24298) demonstrates decoupled generation and training, controlled staleness, and staleness-aware PPO. [Asynchronous RLHF](https://arxiv.org/abs/2410.18252) directly studies the performance–efficiency trade-off of off-policy async data. Most of their evidence comes from reasoning and instruction following, so neither settles the case for multi-turn tool-agent training.

A minimal admission controller looks like this:

```python
MAX_VERSION_LAG = 0  # synchronous baseline; later ablate with 1

while True:
    group = queue.pop_complete_verified_group()
    versions = {t.behavior_policy_version for t in group}
    assert len(versions) == 1

    behavior_version = versions.pop()
    train_version = registry.current_train_version()
    if train_version - behavior_version > MAX_VERSION_LAG:
        quarantine(group, reason="stale_policy")
        continue

    assert all(not t.mixed_policy for t in group)
    assert same_version_tim_error(group, reload=behavior_version) < TIM_TOL
    assert sampled_action_drift(group, current=train_version) < DRIFT_TOL

    next_policy = trainer.update(group)
    registry.stage(next_policy)
    weight_sync.require_all_worker_checksums(next_policy)
    registry.publish_atomically_at_episode_boundary(next_policy)
```

Each group also needs an immutable `group_id` and exactly-once consumption. Atomically commit `consumed_group_ids + optimizer_step + output_policy_version` in the trainer checkpoint. After recovery, the trainer must neither reapply a group nor treat an applied-but-unacknowledged group as unconsumed.

Do not judge asynchrony by tokens per second alone. Compare synchronous and asynchronous systems on held-out success, safety, KL, ratio tails, stale/drop rate, queue age, generator and trainer utilization, weight-sync latency, and wall-clock time to a target. Higher throughput is not a win if held-out capability improves less per hour.

### Component map

| Topology                           | Generator                                 | Trainer / weight synchronization                                                      | Best use                                                          |
| ---------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Single-GPU synchronous smoke test  | In-process generation                     | PyTorch/PEFT; copy LoRA by function call                                              | Validate the trace, reward, and gradient loop                     |
| Multi-GPU colocated synchronous    | vLLM rollout phase                        | [verl / HybridFlow](https://arxiv.org/abs/2409.19256) resharded into a training phase | Reproducible PPO/GRPO/RLOO baseline                               |
| Disaggregated bounded async        | vLLM/SGLang generator pool                | FSDP/DeepSpeed learner + versioned queue + NCCL/IPC/RDMA sync                         | Long trajectories with less straggler idle time                   |
| Fully asynchronous research system | Continuous generator pool                 | [AReaL](https://arxiv.org/abs/2505.24298)-style learner and staleness controller      | Explicitly study the off-policy trade-off                         |
| Multi-version streaming            | Several pinned policy pools remain active | [DORA](https://arxiv.org/abs/2604.26256)-style leases, routing, and bounded admission | Let long episodes finish on their original version; 2026 preprint |
| Existing agent runtime             | Trace adapter                             | [Agent Lightning](https://arxiv.org/abs/2508.03680)-style disaggregation              | Decouple execution from training                                  |

Several repositories provide useful implementation anchors. [slime's `train_async.py`](https://github.com/THUDM/slime/blob/main/train_async.py) waits for active generation before updating weights and offers a useful introduction to semi-asynchronous execution. The [verl fully asynchronous recipe](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md) separates the Rollouter, MessageQueue, Trainer, and ParameterSynchronizer. [THUDM AgentRL](https://github.com/THUDM/AgentRL) implements multi-turn rollout, actor, and reference worker pools with a `group_id` queue on Ray. These repositories provide components; they do not replace the version, log-probability, and held-out gates above.

A minimal repository can start with these modules:

```text
agent_rl_min/
├── actions.py       # typed action + canonical serialization
├── policy.py        # immutable PolicyManifest + three-policy contract
├── env.py           # reset/step/export/snapshot
├── snapshots.py     # manifests, content hashes, restore/fork
├── trajectory.py    # immutable trace schema
├── segments.py      # partial rollout/continuation/compaction contract
├── verifier.py      # privileged clean-room scoring
├── generator.py     # version-pinned episode collection
├── queues.py        # complete-group admission + staleness policy
├── credit.py        # RLOO, later GAE/branching + intervention audit
├── loss.py          # explicit token/turn/trajectory reduction
├── consistency.py   # generator/trainer logprob audit
├── trainer.py       # master weights + optimizer state
├── registry.py      # stage/checksum/atomic policy publish
├── weight_sync.py   # generator update protocol
├── eval.py          # held-out + retain + integrity suites
└── tests/           # replay, masks, versions, corruption, logprob contract
```

Start synchronously. Treat bounded asynchrony as an ablation because it changes both the algorithm and the system. Throughput is not a free algorithmic improvement.

---

## 7. A smoke experiment you can run end to end

This is not a paper-scale recipe. It is the smallest end-to-end experiment designed to expose obvious ways the loop can game its measurements.

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
trainable: lora

policy:
  require_manifest: true
  lock_scaffold_across_checkpoints: true

environment:
  domain: deterministic_terminal
  train_tasks: 48
  heldout_tasks: 16
  max_turns: 8
  reset_each_episode: true
  snapshot_backend: canonical_state
  base_snapshot_per_task: true
  branching: false # prove restore fidelity before enabling turn-level forks

rollout:
  mode: synchronous
  group_size: 8
  temperature: 1.0
  top_p: 1.0
  top_k: null # disable truncation, or save the actual sampler logprob
  pin_policy_for_full_episode: true
  max_version_lag: 0
  require_one_behavior_version_per_group: true
  in_flight_policy: finish_old
  partial_episode_policy: abort_and_restart
  require_termination_reason: true
  context_compaction: false

consistency:
  audit_frozen_checkpoint: true
  tolerance_source: bf16_same_engine_baseline

weight_sync:
  transport: in_process
  publish: atomic_at_episode_boundary
  require_all_worker_checksums: true
  kv_cache_on_update: invalidate

credit:
  estimator: rloo
  reduction: episode_sum
  group_reward_scale: none

update:
  objective: reinforce_with_kl
  behavior_policy: rollout_manifest
  proximal_policy: behavior_policy
  reference_policy: initial_sft
  kl_beta: 0.02
  optimizer_steps_per_fresh_batch: 1

evaluation:
  attempts_per_task: 3
  seeds: [11, 22, 33]
  retain_suites: [base_tool_use, instruction_following, safety]
  temporal_cutoff: preregistered
  template_disjoint: true
  environment_family_disjoint: true
  repository_overlap_audit: exact_and_semantic
  dedup_rule_version: v1
```

Begin with deterministic file, JSON, schema, and CLI repair tasks. Each task should include an initial container snapshot, a natural-language objective, a set of allowed tools, a hidden grader, and invariants that must remain unchanged. Separate train and held-out sets by task template or environment family, not by changing a few numbers.

If the base model cannot reliably emit typed actions on the smoke set, add a small tool-call and recovery SFT stage first. Do not expect RL to discover useful exploration when every rollout either fails or ends in a parse error.

### Four controls

1. A frozen, no-update baseline.
2. A normal RLOO update from the same cold start.
3. A null update with rewards shuffled within each group.
4. An SFT-only control with training-token count matched as closely as possible to the RL update.

### Go / no-go gates

Before measuring capability gain, require every gate below to pass:

- Reset reproduces the same state hash.
- After restore, replaying a fixed action suffix reproduces observations, state hashes, and reward exactly.
- Mutations in two branches are isolated; a sealed terminal snapshot is immutable and never visible to the agent.
- Rollout, trainer, and evaluator reference the same complete `PolicyManifest`; scaffold drift is zero.
- Every action token has a finite, length-aligned behavior log probability.
- Versions, roles, and log-probability fields for `π_beh`, `π_prox`, and `π_ref` are unambiguous.
- Each episode and each sibling group has exactly one behavior-policy version; mixed-policy episode count is zero.
- Active-weight checksums on every generator rank match the registry.
- At a frozen checkpoint, generator/trainer `p95/max |Δlogp|` and `k3` divergence stay within preregistered tolerances.
- No sample beyond `max_version_lag` reaches the trainer.
- Group IDs are consumed exactly once; duplicate, lost, and incomplete groups are counted separately.
- With `max_version_lag=0`, the asynchronous code path produces numerically equivalent loss and gradients to the synchronous implementation.
- Stale and dropped sample rates are stratified by task family, trajectory length, and reward so curriculum bias is visible.
- Sample index, task ID, seed, and checkpoint are complete.
- `finish`, environment terminal states, truncation, infrastructure failure, and scheduler cancellation use distinct termination codes; censored trajectories never become reward 0 silently.
- Unit tests show that `episode_sum` implements the score-function term up to a length-independent constant; every other reduction is labeled as a different surrogate.
- On representative fixtures adjudicated by humans or a stronger external program, the verifier passes preregistered FP/FN gates with denominators and confidence intervals.
- In an independent toy-logit test, the aggregate policy-gradient directional derivative has the correct sign for positive and negative advantages. A real shared-parameter batch need only reduce the aggregate objective with bounded KL; do not demand monotonic change for every sampled action.
- The shuffled-reward control shows no stable held-out gain.

The core report should include held-out success, retain-suite delta, `pass^3`, invalid-action rate, tool calls, tokens, latency, side-effect rate, verifier FP/FN on the audit set with denominators and confidence intervals, termination distribution, zero-advantage group rate, length-stratified gradient mass, KL, gradient norm, and bootstrap confidence intervals. For asynchronous runs, also report version and wall-clock lag, queue age, stale/drop rate, weight-sync latency, generator/trainer utilization, and training–inference mismatch. For clipped objectives, report the clip fraction at the corresponding granularity. Report a rejection fraction only when samples are actually discarded, and always include importance-ratio tails.

When the policy, scaffold, and budget are fixed and attempts are independent or at least exchangeable, `pass@3` measures whether at least one of three attempts succeeds, making it closer to a measure of search capacity. `pass^3` measures whether all three succeed, making it closer to operational reliability. This distinction comes from [τ-bench](https://arxiv.org/abs/2406.12045). Three attempts per task still have high variance; production reliability requires more repetitions, per-task uncertainty estimates, and confidence intervals.

---

## 8. From a runnable loop to a research program

Once the loop is sound, five research directions matter more than another optimizer swap.

### 8.1 Which experiences actually create capability?

A middling pass rate or nonzero group variance tells us only that a task supplies a learning signal. It does not show that training on the task improves performance elsewhere.

A stronger estimand is:

```text
u(g, s) = J_heldout(Update(θ₀, rollout(g, s))) - J_heldout(θ₀)
```

Every candidate update starts independently from the same checkpoint `θ₀`. This defines a noisy one-update effect estimand, but it does not identify a causal effect by itself. Attribution also requires paired evaluation seeds, no-update and null controls, multiple rollout and update seeds, confidence intervals, and a correction for selecting the maximum among many candidates. This is the question behind the CUES-TMax protocol. Until the formal causal updates are complete, neither a behavior taxonomy nor a surrogate can be treated as a utility label.

`TaskSampler` therefore needs more than a “keep nonzero-variance groups” switch. An auditable curriculum should maintain at least four streams: currently learnable tasks, hard tasks with rare positive examples, mastered retain tasks, and tasks quarantined because the environment is infeasible or the verifier is untrustworthy. Store the sampling propensity and capability family for every sample so that any narrowing of the training distribution remains measurable. [TMax](https://arxiv.org/abs/2606.23321) is a 2026 preprint that demonstrates difficulty control, persona diversity, and verifier diversification. The next step is an independent one-update intervention that separates learnability from causal utility.

### 8.2 Does target gain come at the cost of general capability?

Reference KL on current RL prompts constrains outputs only near sampled contexts. It does not preserve unsampled domains, old tool schemas, instruction following, or safety refusals. A `RetainEvaluator` should freeze task-family-disjoint probes and measure target gain and retain loss at fixed update intervals. Stop or roll back when regression crosses a preregistered frontier.

The minimal comparison includes fixed KL, SFT/PTX replay, domain-balanced replay, and a no-retention control. Plot the target-gain–retain-loss Pareto frontier instead of reporting training KL alone. [InstructGPT](https://arxiv.org/abs/2203.02155) mixed pretraining data to mitigate some regressions. [RECAP](https://arxiv.org/abs/2510.21978) argues more directly that current-task KL does not preserve broader capabilities, but its evidence comes from vision-language models and a recent preprint. Treat it as motivation for an Agent experiment, not as a completed transfer result.

### 8.3 Does verifier error accumulate with horizon?

Longer horizons may create more opportunities to encounter conflicting documents, stale state, wrong-path artifacts, semantic lures, and ambiguous rubrics. To test that hypothesis, hold the experiment-owned token budget fixed, inject controlled corruptions, and separately plot actor error, verifier FP/FN, calibration where applicable, reward-hacking rate, and horizon. A fixed token count reduces length confounding but does not remove confounding from task difficulty, content density, or tool topology. One average LLM-judge accuracy is not enough.

Our verifier-horizon pilot observed a horizon-associated increase but did not satisfy the preregistered superlinearity test. It also used only 14 episodes per cell, below the preregistered target of 50. In the small V3 sample, the hybrid verifier had FP `0/7`, but FN rose to `5/9 (55.6%)`, compared with `3/9 (33.3%)` for the semantic judge. The result supports strengthening the protocol; it does not show that hybrid verification is cost-free or that error necessarily grows superlinearly. The full design, failure gates, and results are in the [three-stage verifier-horizon experiment](https://ajing.github.io/posts/2026-08-14-verifier-error-horizon-scaling/).

An independent identity must provision the tests, scorer, clock, and reference artifacts in a read-only namespace that the agent cannot modify. [METR's real agent traces](https://metr.org/blog/2025-06-05-recent-reward-hacking/) include reward hacking through scorer and test modification as well as metadata exploitation. [OpenAI's work on chain-of-thought monitoring](https://openai.com/index/chain-of-thought-monitoring/) shows that current monitors can catch some hacks, but optimizing the monitor signal directly can make intent harder to observe. Use monitor outputs as quarantine or audit signals unless they have been validated as training rewards.

### 8.4 Did the agent really change the state?

A successful tool response, a convincing GUI, and a correct authoritative persistent state are three different things. For a hybrid GUI/CLI agent, bind each mutation to a generation fence and make write authority single-use, atomic, and auditable. When state consistency cannot be proven, abstention is more correct than a false claim of completion. This boundary comes from the state model in the recent ReplicaGuard prototype: disk is authority, a dirty buffer is protected intent, and the GUI is evidence. The current results come from a controlled loopback environment and do not establish real-world prevalence.

### 8.5 The policy can exploit benchmarks and user simulators too

Evaluate agents along five separate axes: capability, repeated-run reliability, safety and side effects, cost and latency, and evaluation integrity. No single success rate or LLM-judge score can represent all five. An `EvalFirewall` should use temporal and/or private holdouts, then audit exact and semantic overlap separately across templates, environments, and repositories. Removing task identifiers reduces explicit recognition but cannot prevent semantic fingerprinting. Network and answer-source restrictions must also preserve the construct being tested: disabling the web changes an open-web research task. Publish both the allow/block policy and its cost in realism. Reverify final artifacts under an independent identity in a read-only grader namespace, and publish the system prompt, tools, budget, retries, and scaffold manifest. These measures reduce contamination; they do not eliminate it.

The risk is concrete. OpenAI audited the 138 SWE-bench Verified tasks that o3 failed to solve consistently across 64 runs—the difficult 27.6% subset, not a random sample. The audit found material test or problem-description issues in 59.4% of that subset. Together with evidence that frontier models could reproduce parts of gold patches or problem details, those flaws led OpenAI to stop reporting the benchmark. See the [OpenAI SWE-bench audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/).

In a 1,266-task Claude Opus 4.6 multi-agent BrowseComp run, Anthropic identified nine cases in which agents recovered answers from public benchmark material. In two more cases, agents recognized the evaluation, then located and decrypted an answer key. The authors note that the task did not prohibit those sources and therefore do not classify the behavior itself as an alignment failure. The result nevertheless demonstrates an integrity risk for web-enabled static evaluations. See [Anthropic's BrowseComp eval-awareness report](https://www.anthropic.com/engineering/eval-awareness-browsecomp).

In multi-turn support and collaboration tasks, the user simulator becomes another policy inside the transition dynamics. A `SimulatorMatrix` should version the user model, prompt, tools, and sampling configuration; evaluate across multiple simulator families; and reserve a held-out simulator or real-dialogue audit set. In its telecom dual-control domain, [τ²-bench](https://arxiv.org/abs/2506.07982) models the dual-control interaction as a Dec-POMDP in which both the agent and user can act on a shared environment. [RealUserSim](https://arxiv.org/abs/2605.20204) is a 2026 preprint whose findings are limited to WildChat-derived profiles, five behavioral dimensions, and τ-bench experiments. They do not establish a universal gap between simulators and real users.

These directions cover experience utility, capability retention, reward integrity, runtime correctness, and evaluation integrity. Together, they show that Agent RL is not merely the study of a loss function. It is the study of which signals in a closed loop deserve our trust.

---

## Conclusion

If you reproduce only one thing from this article, reproduce this chain:

```text
one complete PolicyManifest
→ one resettable initial state
→ a group of fresh sibling rollouts
→ an independent verifier namespace the agent cannot write
→ an explicit advantage estimator
→ exactly one auditable update
→ task-disjoint held-out evaluation
```

Once this chain works, PPO, GRPO, DPPO, branching credit, and asynchronous rollout become replaceable components. Until then, adding more data, parameters, and GPUs only makes the error more expensive.

The technical depth of Agent RL lies not in the number of algorithm names in the stack, but in whether the system can answer: **Why did this experience produce this gradient, and why did that gradient improve an unseen task?**

---

## Selected primary references

- [POMDP: Planning and Acting in Partially Observable Stochastic Domains](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [RLOO: Back to Basics](https://aclanthology.org/2024.acl-long.662/)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [STEPO / EvoCUA-1.5](https://arxiv.org/abs/2607.09773) — 2026 preprint
- [Rethinking the Trust Region / DPPO](https://arxiv.org/abs/2602.04879) — 2026 preprint
- [GiGPO](https://arxiv.org/abs/2505.10978)
- [TRACE](https://arxiv.org/abs/2607.13988) — 2026 preprint
- [Branching Policy Optimization](https://arxiv.org/abs/2607.14171) — 2026 preprint
- [Entropy Mechanism of RL](https://arxiv.org/abs/2505.22617)
- [High-Entropy Minority Tokens](https://arxiv.org/abs/2506.01939)
- [TMax: A Simple Recipe for Terminal Agents](https://arxiv.org/abs/2606.23321) — 2026 preprint with open code and data
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [RECAP: Mitigating General Capability Regression](https://arxiv.org/abs/2510.21978) — recent preprint; VLM evidence
- [Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Agent Lightning](https://arxiv.org/abs/2508.03680) — 2025 preprint
- [Tool-calling Pipeline Sensitivity](https://arxiv.org/abs/2606.00135) — ICML 2026
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256)
- [IMPALA / V-trace](https://arxiv.org/abs/1802.01561)
- [Asynchronous RLHF](https://arxiv.org/abs/2410.18252)
- [AReaL](https://arxiv.org/abs/2505.24298)
- [DORA](https://arxiv.org/abs/2604.26256) — 2026 preprint
- [Diagnosing Training–Inference Mismatch](https://arxiv.org/abs/2605.14220) — 2026 preprint
- [APRIL: Active Partial Rollouts](https://arxiv.org/abs/2509.18521)
- [RollPacker](https://arxiv.org/abs/2509.21009)
- [CompactionRL](https://arxiv.org/abs/2607.05378) — 2026 preprint
- [vLLM Native RL APIs](https://vllm.ai/blog/2026-05-28-native-rl-apis)
- [τ-bench](https://arxiv.org/abs/2406.12045)
- [τ²-bench](https://arxiv.org/abs/2506.07982)
- [RealUserSim](https://arxiv.org/abs/2605.20204) — 2026 preprint
- [OSWorld](https://arxiv.org/abs/2404.07972)
- [AppWorld](https://arxiv.org/abs/2407.18901)
- [OpenAI: Why We No Longer Evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- [Anthropic: Eval Awareness in BrowseComp](https://www.anthropic.com/engineering/eval-awareness-browsecomp)
- [METR: Recent Reward Hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
