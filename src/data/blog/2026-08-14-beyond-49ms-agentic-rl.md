---
author: Jing Lu
pubDatetime: 2026-08-14T18:00:00Z
title: "Beyond 49 ms: Where VM Resume Latency Actually Goes in Agentic RL"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Reinforcement Learning
  - Systems
description: "A measurement-first look at fast sandbox resume, speculative tool execution, and action-observation co-speculation—and why time-to-useful-work matters more than boot time."
---

An agent can resume a VM in 49 milliseconds and still spend seconds waiting to make progress.

That is the systems problem I have been trying to characterize. In an agentic rollout, “the sandbox is running” is not the same event as “the model has received a useful observation.” Between those two points sit routing, restore, guest readiness, tool execution, output transport, and sometimes a second round of model scheduling.

The distinction matters even more now. Recent systems do not merely make sandboxes faster. They predict which sandbox an agent will need, launch likely tool calls before the model finishes emitting them, and even let the model continue decoding against a predicted observation in a speculative branch.

The short version is:

> **VM resume is becoming a small, well-engineered primitive. The next bottleneck is the serialized boundary between model generation, environment execution, and the next useful observation.**

I built a benchmark harness to measure most of that boundary—from a resume request through the first validated useful command. The harness exists; the formal KVM results do not. Policy ingestion and next-model scheduling also remain outside the current harness. This post is therefore not a victory lap around an unmeasured “49 ms” result. It is a claim-bounded account of what that number means, what it excludes, and how the newest speculative systems try to hide the rest.

## The latency is a pipeline, not a boot time

A simplified agent step looks like this:

```text
LLM emits action
      │
      ▼
route / acquire sandbox
      │
      ▼
resume VM and reconnect process
      │
      ▼
execute tool or shell command
      │
      ▼
return a usable observation
      │
      ▼
schedule LLM and continue rollout
```

Without overlap, the critical path is approximately:

```text
Tstep = Tdecode-action
      + Tacquire
      + Tresume
      + Ttool
      + Tobservation
      + Tdecode-next
```

Most “fast sandbox” claims optimize one term. They may do it extremely well, but the user experiences the sum.

This is also why environment latency behaves differently from ordinary inference latency. A model server can batch tokens from many requests. An interactive environment often carries mutable state, file-system changes, long-running processes, and external side effects. It cannot always be replaced by another warm replica. The runtime has to preserve the identity of the world in which the agent is acting.

I previously discussed partial rollouts and asynchronous environment scheduling in [From Long CoT to Agent Swarms](https://ajing.github.io/posts/2026-07-18-from-long-cot-to-agent-swarms-kimi-rl/). Here I will stay on the systems side of that boundary: resume, readiness, speculation, verification, and rollback.

## What does “under 50 ms” actually measure?

[AgentENV](https://github.com/kvcache-ai/AgentENV), the sandbox system used by Kimi K3, currently reports snapshot-backed boot or resume below 50 ms and pause below 100 ms. The more precisely scoped [Kimi K3 technical report](https://arxiv.org/abs/2607.24653) reports a minimum 49 ms resume and 133 ms incremental checkpoint in its deployment. AgentENV combines Firecracker, OverlayBD, ublk, and incremental memory and file-system snapshots.

That is impressive. It is also not a complete end-to-end latency definition.

At the commit I pinned for my harness, AgentENV's official [snapshot benchmark](https://github.com/kvcache-ai/AgentENV/blob/8174bd9674806917a2c0e75d26a30b409459c51d/benches/snapshot_benchmark.rs) calls the Firecracker sandbox resume path directly. That path waits for Firecracker startup, `envd` readiness, and guest initialization, so it is more meaningful than timing the hypervisor process alone. But it remains a node-local runtime operation. It does not traverse the full client, HTTP gateway, scheduler, placement path, persistent metadata layer, run a user command, or return that command's output to the policy.

There is a second measurement trap. The official concurrent benchmark's ten reported positions are averages across 50 workers, each running ten iterations: 500 resume operations compressed into ten aggregate values. Those values cannot be used to infer a per-request p95 or p99. Tail latency requires retaining every request sample.

That is not a flaw in the benchmark. It is a boundary around its claim.

I find it useful to separate five clocks:

| Clock         | Start                 | Stop                                                  | What it answers                                                |
| ------------- | --------------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| Backend-ready | Runtime enters resume | Node-local resume returns after `envd` initialization | How fast is the local restore primitive?                       |
| Resume RPC    | Client sends request  | Client receives success                               | What do routing and control-plane work add?                    |
| Service-ready | Client sends request  | Guest agent accepts work                              | Can the resumed sandbox actually receive a command?            |
| TTFC          | Client sends request  | First command completes                               | How soon can the sandbox do anything useful?                   |
| TTUW          | Client sends request  | Useful-work oracle validates the result               | When has the restored sandbox produced correct, useful output? |

Here TTFC means **time to first command**, not time to first character. TTUW means **time to useful work**. In the current harness, TTUW stops when a correctness oracle validates the command result. Observation ingestion, policy scheduling, and the first token of the next model turn require an additional clock.

The gap can be small on a quiet single-node system. Under concurrent resume, placement pressure, cache misses, guest reconnects, large snapshots, and rollout scheduling, it may not be.

The last row matters.

## What I built, and what I have not measured

I built a small project called `beyond-49ms` to turn those clocks into reproducible measurements. The current harness includes:

- per-request samples and structured traces;
- separate backend, RPC, TTFC, and TTUW timing stages;
- serial, synchronized-barrier, and open-loop workload drivers;
- Linux, KVM, and Firecracker preflight checks; and
- a wrapper around the pinned official AgentENV snapshot benchmark.

The workload modes are deliberate. A serial median can make almost any warm system look good. A synchronized barrier exposes herd behavior. An open-loop arrival process shows whether the service accumulates a queue when offered load exceeds sustainable resume throughput.

The missing result is equally important: **I have not yet produced the formal AgentENV KVM dataset.** My local machine is macOS, and the cloud node available for the first pass does not expose `/dev/kvm`. The repository's baseline report therefore still says “not yet measured.” In the current local checkout, twenty unit tests pass, and local `/bin/echo` workloads exercise the recorder, concurrency drivers, and analysis path. They are smoke tests, not VM data.

Current project status:

| Component                              | Status           |
| -------------------------------------- | ---------------- |
| Measurement schema and traces          | Implemented      |
| Serial, barrier, and open-loop runners | Implemented      |
| AgentENV benchmark wrapper             | Implemented      |
| Real KVM baseline                      | Not yet measured |
| Partial-rollout simulator              | Planned          |
| Resume fault injection                 | Planned          |
| Learned pause/resume policy            | Planned          |

This means the project supports a measurement program. It does not yet support a performance claim.

That negative evidence changed the direction of the work. Instead of asking only “how do I shave another 10 ms from resume?”, the more useful question became:

> Which parts of the action-observation gap must be made faster, and which parts can be moved off the critical path entirely?

## Layer 1: make the execution substrate genuinely fast

Snapshot resume remains the foundation. If a branch takes hundreds of milliseconds or seconds to materialize, higher-level speculation will have a narrow break-even range.

AgentENV restores a persistent sandbox rather than provisioning a generic container from scratch. This is especially useful in agentic RL because a rollout environment may contain installed packages, modified files, shell history, background processes, and a long sequence of earlier actions. Reconstructing that state from a textual transcript is both slower and less faithful than resuming the state itself.

Other runtimes are converging on a similar target. Tencent's [CubeSandbox](https://github.com/TencentCloud/CubeSandbox) reports a fully serviceable sandbox below 60 ms and supports automatic pause and resume. Its published numbers are vendor-run rather than results from my harness, so I treat them as evidence of the design direction, not as a head-to-head comparison.

Fast resume helps in three places:

1. **Cold allocation:** reduce the delay before a new trajectory can act.
2. **Idle suspension:** stop paying for CPU and memory while the model is thinking.
3. **Branching:** cheaply create isolated worlds for candidate or speculative actions.

The third use is the most interesting. Once resume and fork are cheap enough, a sandbox stops being only a place where the “real” trajectory runs. It becomes a unit of speculation.

## Layer 2: predict demand and prewarm the sandbox

The safest speculation does not guess an observation. It guesses that an environment will soon be needed.

[SpecBox](https://arxiv.org/abs/2607.23933) predicts Docker sandbox demand while the model is still generating. It uses lightweight keyword signals and streaming semantic embeddings, prefetches dependencies through a graph, caches reusable results, and avoids unnecessary data copies. At QPS 20 in its single-server experiment, P99 end-to-end latency fell from 257.2 seconds for reactive provisioning to 88.7 seconds, or roughly a 2.9× speedup. This is predictive prewarming, not microVM snapshot restore.

The key operation is overlap:

```text
Without prewarming:
model generation ──────────┐
                           └─ allocate + prepare ─ tool

With prewarming:
model generation ─────────────── tool
             └─ allocate + prepare ─┘
```

If the prediction is wrong, the system discards an unused sandbox. The canonical trajectory has not consumed fabricated data, and the action has not produced an external side effect. The cost is wasted capacity, not corrupted state.

This makes intent-based prewarming a natural first deployment step. It also means hit rate alone is the wrong objective. A policy should prefer predictions that hide a long delay at acceptable resource cost. A modestly probable 500 ms setup may be more valuable than a highly probable 20 ms setup.

## Layer 3: predict the action and start real execution early

The next layer predicts the concrete future action.

[Speculative Actions](https://arxiv.org/abs/2510.04371) uses a fast model to predict later agent actions, executes those actions in parallel, and reuses a result only when the main model's eventual action matches. The updated paper reports next-action prediction accuracy up to 55% and end-to-end latency reduction up to 20%.

[PASTE](https://arxiv.org/abs/2603.18897) pushes the idea into an agent-serving scheduler. It predicts concrete tool invocations before the LLM finishes generating them, isolates speculative results, and coordinates the returning model session with tool completion. Its current version reports up to a 43.5% reduction in average task completion time and a 1.8× improvement in observed tool latency. These are the revised v3 results; the earlier 48.5% headline is stale.

The correctness boundary is important. Neither system has to pretend that a guessed tool result is true. It can run the **real tool call** early in an isolated context, then attach the real result if the predicted and emitted calls match.

That turns part of the sum into a maximum:

```text
serial:      Tgenerate + Ttool
speculative: max(Tgenerate, Ttool) + Tverify
```

This works best when actions are predictable, tools are slow, arguments become stable early, and isolation is cheap. It works poorly when exact arguments change late, the action has an irreversible external effect, or verification costs nearly as much as waiting.

## Layer 4: predict the observation and let the model continue

There is a more aggressive idea: do not wait for the environment result before continuing the rollout. Predict the result, give it to a speculative copy of the model, and validate the branch when real execution finishes.

This is the work I had been trying to remember. The closest and now most direct formulation is [AOSpec: Action and Observation Co-Speculation for Low-Latency Agent Serving](https://arxiv.org/abs/2608.00881), released in August 2026.

AOSpec jointly speculates actions and observations. Its Expected Value Decoding policy ranks observation candidates using both their probability and the tool time they could hide. For outputs that depend on mutable environment state, it launches the target action in an isolated fork. Its Joint Action-State Verification checks both the eventual action and the state from which that action originated before reusing speculative work.

Conceptually:

```text
canonical branch:   action ───── real tool ───── observation ─ continue
                                  │                 │
speculative branch: predicted observation ─ model continues
                                  │
                         commit only after
                       action + state verification
```

On Terminal-Bench, across four harnesses, five actor models, and five serving speeds, AOSpec reports mean end-to-end latency reductions from 11.8% to 32.5%, with P99 reduction up to 42.8%. At 1 ms time per output token, joint action-observation speculation saved 32.5% on average. On an unseen SWE-bench transfer setting at the same serving speed, it reported 18.9% savings versus 4.2% for the strongest baseline.

Those numbers answer the remembered question: **yes, the idea now appears to work across more than a toy trace.** But it does not work by blindly returning a hallucinated result to the live agent. It works by moving the continuation into a shadow branch and retaining serial semantics at commit time.

That distinction matters.

[DyMo](https://arxiv.org/abs/2506.02918) is related but solves a different problem. It trains an internal dynamics model to generate tool calls and predict resulting states, then uses those predictions for self-verification and candidate selection. It shows that learned environment models can improve action selection, but it is not primarily an end-to-end serving system for committing predicted observations. Its reported true-negative rate below 50% is also a useful reminder: a model that is good enough to rank candidates is not automatically trustworthy enough to mutate canonical state.

[Qwen-AgentWorld](https://arxiv.org/abs/2606.24597) goes further toward a learned simulator: it trains a language world model on more than ten million interaction trajectories to predict full textual observations across MCP, search, software engineering, terminal, mobile, web, and operating-system tasks. But its own ablation contains the warning label. On Tool Decathlon, uncontrolled simulated RL moved the score from 32.4 to 31.5; adding control raised it to 36.1. Predicted observations can be useful training or planning inputs. Ungrounded observations can also train the policy in the wrong world.

## Layer 5: make mismatch and recovery cheap

Speculation becomes practical only when a wrong branch is cheap to discard.

[DeltaBox](https://arxiv.org/abs/2605.22781) treats rollback as a change-tracking problem. Its DeltaFS and DeltaCR components capture file-system and process deltas instead of repeatedly copying a full sandbox. On its SWE-bench and RL microbenchmarks, the paper reports 14 ms checkpoint and 5 ms rollback.

[Crab](https://arxiv.org/abs/2604.28138) makes checkpointing semantics-aware. It observes that more than 75% of agent turns contain no recovery-relevant state, uses an eBPF-based inspector to choose checkpoint granularity, and overlaps checkpoint work with model waiting time. The paper reports 100% recovery correctness, an 87% reduction in checkpoint traffic, and runtime within 1.9% of a fault-free execution.

These systems are usually framed as fault-tolerance mechanisms. They also supply the transaction primitive that speculation needs:

```text
fork → execute → verify → commit
                   └────→ discard / roll back
```

Fast resume without cheap discard creates a fast way to start work. Fast resume with cheap discard creates a substrate for parallel counterfactual work.

## Branching can improve the learning signal too

Fast snapshots are not only a serving optimization. [Branching Policy Optimization](https://arxiv.org/abs/2607.14171) snapshots a high-entropy intermediate state, forks several sibling actions from exactly that state, and uses their returns to construct a step-level baseline.

The paper reports 3.6–6.1 percentage-point success improvements over matched-compute GRPO and RLOO baselines, roughly half the gradient-norm variance, and 38% fewer policy updates to reach the best baseline's performance. Its measured snapshot overhead ranged from 42 ms on WebShop to 1.92 seconds for a SWE-bench overlayfs environment.

This is promising evidence that resumable state can improve both systems utilization and credit assignment. It is not yet an AgentENV result: WebShop and ALFWorld use simulator state, while the SWE-bench path uses Docker overlayfs rather than Firecracker microVM snapshots.

## The five mechanisms attack different gaps

The reported results below are study-specific. They use different tasks, hardware, baselines, and latency definitions. They should not be read as a leaderboard.

| System              | What is anticipated        | What starts early                       | Correctness boundary                 | Reported result                                 |
| ------------------- | -------------------------- | --------------------------------------- | ------------------------------------ | ----------------------------------------------- |
| AgentENV            | Nothing                    | Snapshot resume                         | Restore the real sandbox             | Boot/resume below 50 ms                         |
| SpecBox             | Sandbox demand             | Allocation and dependencies             | Discard unused capacity              | Roughly 2.9× faster P99 at QPS 20               |
| Speculative Actions | Future action              | Real tool execution                     | Reuse only on action match           | Up to 20% latency reduction                     |
| PASTE               | Concrete tool invocation   | Tool plus LLM rescheduling              | Isolated result, match before use    | Up to 43.5% lower average task time             |
| DyMo                | Action and resultant state | Candidate generation                    | Use prediction for self-verification | Higher pass@k-style selection accuracy          |
| AOSpec              | Action and observation     | Forked execution and model continuation | Joint action-state verification      | 11.8–32.5% mean E2E reduction                   |
| DeltaBox / Crab     | Recovery-relevant state    | Checkpoint and branch support           | Commit, discard, or recover          | Millisecond rollback / lower checkpoint traffic |

The stack is cumulative, not mutually exclusive:

```text
fast resume
    └─ intent-based prewarming
          └─ speculative real action
                └─ speculative observation
                      └─ cheap verification and rollback
```

Each deeper layer hides more serialized latency and assumes a stronger correctness mechanism.

## The safety rule: speculation may read ahead, not commit ahead

The clean implementation rule is:

> **A predicted observation can advance a shadow rollout. It cannot become an authoritative fact until the corresponding action and origin state have been verified.**

I would enforce five invariants:

1. **Branch isolation.** Speculative file, process, and memory changes live in a fork.
2. **Action identity.** Tool name, normalized arguments, and relevant environment inputs must match.
3. **State identity.** Verification must include the snapshot or state version from which the action ran.
4. **Side-effect control.** Irreversible network, payment, messaging, or production actions are not speculated without an idempotency or transaction layer.
5. **Deterministic fallback.** A mismatch discards the branch and resumes from the last verified state.

Verifying only the action is insufficient. The same `pytest` command can produce a different result after one file edit. The same SQL query can observe a different database version. The same browser action can target a changed page.

The pair `(action, origin state)` is the reusable unit.

## Why this matters more for agentic RL

In a single interactive session, a 200 ms pause is annoying. In RL, it is multiplied by many environments, many turns, and a long tail of stragglers.

Consider a training system running thousands of rollouts. Some trajectories are waiting for model tokens, some for sandboxes, some for package installation or tests, and some are already complete. A synchronized iteration waits for the slow tail. Partial-rollout training can reduce that barrier, but environment state still has to survive pauses and resume when the rollout is scheduled again.

Fast pause/resume therefore changes more than latency. It changes scheduling economics:

- idle environments can release resources while the policy is decoding;
- active trajectories can be interleaved without reconstructing their state;
- slow tool calls can run concurrently with predicted future work;
- failed or low-value branches can be abandoned cheaply; and
- capacity can be assigned to trajectories by expected progress rather than by permanent reservation.

This is where VM work meets RL systems work. The environment runtime supplies persistence and branching. The scheduler decides when a branch is worth keeping alive. The learned policy estimates what will happen next. A verifier preserves semantics when the estimate is wrong.

## A break-even rule for speculation

Prediction accuracy is not enough. A speculative policy should maximize expected latency hidden under a resource and correctness budget.

A simple decision rule is:

```text
speculate when

P(hit) × Lhidden
    > Cverify + (1 - P(hit)) × Cmiss + λ × Cwaste
```

Where:

- `P(hit)` is the probability that action, arguments, observation, and origin state will validate;
- `Lhidden` is the serial latency moved behind useful model work;
- `Cverify` is the validation and commit cost;
- `Cmiss` is any added critical-path penalty after a mismatch;
- `Cwaste` is compute, memory, I/O, and branch-cleanup work; and
- `λ` converts resource waste into the scheduler's current scarcity price.

This explains why AOSpec weights probability by estimated tool time. A very likely prediction is not valuable if it hides almost nothing. A less likely prediction may be worthwhile if it overlaps a multi-second test suite and the sandbox fork is cheap.

The same rule can govern pause decisions. Keeping a sandbox resident avoids future restore latency but consumes scarce memory. Pausing it saves capacity but introduces a resume penalty. The optimal threshold changes with queue depth, snapshot size, expected think time, and cache locality. A fixed idle timeout is only a baseline.

## The benchmark I want to run

The next experiment should measure the whole stack, not one attractive microbenchmark.

### Workload matrix

| Dimension       | Values                                                           |
| --------------- | ---------------------------------------------------------------- |
| Lifecycle       | cold create, snapshot resume, warm resident, predicted prewarm   |
| Concurrency     | 1, 8, 32, 128, saturation                                        |
| Arrival pattern | serial, barrier burst, open loop                                 |
| Snapshot        | tiny shell, package-heavy, active processes, dirty file system   |
| Tool            | no-op, shell, test suite, package install, long-running service  |
| Speculation     | off, sandbox only, action, action + observation                  |
| Outcome         | hit, argument mismatch, state mismatch, timeout, runtime failure |

### Metrics

I would report:

- backend-ready, resume RPC, service-ready, TTFC, and TTUW separately;
- policy-ingestion delay and time to the first token of the next model turn;
- p50, p95, p99, and maximum—not only the median;
- resume throughput and queueing delay under offered load;
- CPU, memory, I/O, snapshot bytes, and branch cleanup time;
- speculation hit rate by tool and horizon;
- latency saved per correct prediction;
- resource waste and tail penalty per mismatch;
- recovery correctness under injected failures; and
- cost per useful rollout step.

### Pass conditions

The experiment should have explicit gates:

1. The resumed environment passes a state-equivalence suite.
2. TTUW remains stable under the target concurrency, not merely concurrency one.
3. P99 does not collapse under barrier bursts.
4. Speculation never commits an action-state mismatch.
5. The end-to-end saving exceeds verification and cleanup overhead.
6. A learned pause or speculation policy beats fixed-timeout and no-speculation baselines on held-out traces.

Until those conditions are measured, “49 ms” is a useful component result, not an answer to agent turnaround time.

## What I would build next

The implementation sequence should follow the correctness gradient.

First, finish the real KVM baseline and expose every timestamp in a single trace. This establishes the difference between backend-ready and useful-work latency.

Second, add intent-based sandbox prewarming. It is the lowest-risk way to convert serial setup into overlap.

Third, introduce forked speculative execution for deterministic, reversible tools such as repository search, compilation, and tests. Validate normalized action arguments and the origin snapshot before attaching the result.

Fourth, add predicted observations only as shadow-rollout inputs. Measure not just top-1 accuracy but verified continuation length: how many model tokens or future actions survive before a mismatch invalidates the branch?

Finally, train a scheduling policy over real traces. Its action space should include keep-resident, pause, resume, prewarm, fork-and-execute, and abstain. Its reward should include latency, resource cost, and mismatch cleanup—not task accuracy alone.

This sequence makes every stage useful even if the more ambitious observation model underperforms.

## Bottom line

The original question was how to spin up VMs faster for agentic workloads. Snapshot-backed runtimes now make a credible sub-50-ms case for the local resume primitive.

The more important question is what happens after that primitive returns.

Fast sandbox systems remove cold-start latency. Prewarming moves setup under model generation. Speculative actions move real tool work under generation. Observation co-speculation lets a shadow rollout cross the tool boundary before the real result returns. Cheap checkpoint and rollback make the inevitable misses affordable.

The systems principle is simple:

> **Measure time to useful work, speculate only inside an isolated branch, and commit only after action-state verification.**

My benchmark harness is ready to test the first part, but the formal KVM dataset is still missing. AOSpec and the surrounding work make the next phase clearer: the target is no longer just a faster VM. It is a latency-aware transactional runtime for agents.

## Primary sources

- [AgentENV: sandbox infrastructure for agentic AI](https://github.com/kvcache-ai/AgentENV)
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [SpecBox: Speculative Sandbox Scheduling for Efficient LLM Agent Serving](https://arxiv.org/abs/2607.23933)
- [Speculative Actions: A Lossless Framework for Faster Agentic Systems](https://arxiv.org/abs/2510.04371)
- [PASTE: Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving](https://arxiv.org/abs/2603.18897)
- [AOSpec: Action and Observation Co-Speculation for Low-Latency Agent Serving](https://arxiv.org/abs/2608.00881)
- [World Modelling Improves Language Model Agents (DyMo)](https://arxiv.org/abs/2506.02918)
- [Qwen-AgentWorld: Language World Models for General Agents](https://arxiv.org/abs/2606.24597)
- [DeltaBox: Scaling Stateful AI Agents with Millisecond-Level Sandbox Checkpoint/Rollback](https://arxiv.org/abs/2605.22781)
- [Crab: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes](https://arxiv.org/abs/2604.28138)
- [Branching Policy Optimization](https://arxiv.org/abs/2607.14171)
