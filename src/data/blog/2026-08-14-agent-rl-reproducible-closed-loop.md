---
author: Jing Lu
pubDatetime: 2026-08-14T19:35:03-07:00
title: "从一次工具调用到一次策略更新：Agent RL 的可复现最小闭环"
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
description: "一份可复现的 Agent RL 技术蓝图：环境 snapshot、verifier、credit assignment、三类 policy、generator–trainer async、partial rollout，以及可运行的 smoke test。"
---

> Agent RL 真正困难的地方，不是选 PPO 还是 GRPO，而是证明一段环境交互如何变成了可信的梯度。

副标题：**三份 contract、一个因果测试，以及一套可以从单机 smoke 开始搭建的 reference implementation。**

很多 Agent RL 介绍都会从算法名开始：PPO、GRPO、RLOO、DAPO。这样很容易把一个系统问题写成 loss function 综述。

但如果一条 trajectory 最终得了 1 分，我们仍然要回答五个问题：

1. Agent 改变的究竟是哪一个环境状态？
2. 这个 1 分来自独立 verifier，还是 agent 自己声称“完成了”？
3. 成功应该归因于哪一个 turn、tool call 或 token？
4. 这些动作究竟由哪个 `π_beh` 采样，它与当前被更新的 `π_train` 相差多远？
5. 比较 checkpoints 时，prompt、tools、memory、sampler 和执行 scaffold 是否保持不变？

任何一个问题答不清，训练曲线都可能上涨，但我们不知道模型学到了什么。

本文不做一份 Agent RL 百科。我只搭一个最小、可运行、可审计的闭环：

```text
Task sampler → resettable environment → agent rollout
     ↑                                  ↓
held-out eval ← policy update ← credit ← verifier
```

全文使用同一套 reference implementation。读完后，你应该可以把 terminal、browser 或 API agent 接进来，先跑通一个小规模实验，再决定是否需要更复杂的 credit assignment 和分布式系统。

整个系统可以压缩成三份合同：

- **Transition Contract**：环境里到底发生了什么？
- **Verification Contract**：终态为什么应该得到这个 reward？
- **Update Contract**：这个 reward 如何对应到 action token 和 policy update？

---

## 1. 先确定训练对象：Agent 是一个 POMDP policy

Agent 与普通单轮 RLVR 的关键差别，是动作会改变后续可见的世界。

一条轨迹可以写成：

```text
τ = (o₀, a₀, o₁, a₁, …, o_T)
```

真实状态 `s_t` 通常不可完全观察：数据库里可能有 agent 没读到的值，GUI 可能落后于磁盘，用户意图也可能只被部分表达。因此 policy 实际依赖的是历史：

```text
πθ(a_t | h_t),  h_t = (o_≤t, a_<t)
```

这更接近 [POMDP](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)。一次 LLM call 还可以被视为一个持续时间不固定的 macro action，因此长程 agent 往往也带有 SMDP/options 的味道。

### 最重要的工程选择：动作粒度

对 tool-using agent，环境层的动作通常是整个 tool call：

```json
{ "kind": "tool", "name": "read_file", "args": { "path": "src/app.py" } }
```

但 policy gradient 最终落在生成这段动作的 token 上：

```text
log πθ(a_t | h_t)
= Σ_{k ∈ action_mask_t} log πθ(y_{t,k} | h_t, y_{t,<k})
```

`tool result`、system prompt、用户文本和其他 agent 的输出不是当前 policy 的动作，必须从 loss 中 mask 掉。否则模型会被训练去“预测环境返回了什么”，而不是学习应该做什么。

安全也不宜全部塞进一个可相互抵消的 scalar：

```text
maximize   E[R_task]
subject to E[C_unsafe] ≤ d
```

严重越权、grader tampering 和不可逆副作用更适合作为 hard gate 或 constrained objective，而不是允许它们被高 task reward 抵消。

### Policy boundary：训练的不是一个裸 checkpoint

对 Agent，真正被部署和评测的 policy 不只是模型参数：

```text
Π = (weights, system prompt, chat template, tool schemas,
     parser/canonicalizer, memory/compactor, retry/budget, sampler)
```

只改 system prompt、多轮 history propagation 或 tool-call template，就可能显著改变结果；因此把 scaffold 变化算成“模型能力提升”是混淆。[一项系统研究 tool-calling evaluation pipeline 的工作](https://arxiv.org/abs/2606.00135)也显示，seed、system prompt、多轮模板和 history propagation 都会改变评测表现。

最小实现应增加一个不可变的 `PolicyManifest`：

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

所有 rollout、update 和 eval 都引用这个 manifest，而不是只写一个 checkpoint name。最小敏感性测试也要成对进行：固定权重只换 scaffold，测量 pipeline variance；固定 scaffold 比较 checkpoints，才把差异归到训练更新。

---

## 2. Transition Contract：Environment 不是 prompt wrapper

一个能训练的环境至少需要四个能力：精确 reset、执行 typed action、导出权威状态、独立验证结果。如果要做局部 credit，还需要 snapshot/restore。

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

`Verifier` 被刻意放在 `AgentEnv` 外面。Agent 可以观察环境，但不能控制最终评分进程。Scorer、hidden tests 和 reference artifacts 应由独立身份重新 provision 到 immutable、read-only、agent 无写权限的 namespace，再对最终 artifact 复验；不能相信 agent 自己跑出的 `pytest passed`。一个新的 rootless container 可以是实现层，但它本身不是完整安全边界；高风险任意代码仍需更强隔离和最小权限。

### Snapshot 是状态契约，不只是“保存一下”

最基础的 outcome-RL 只要求每个 episode 能从同一初始状态 reset；但只要需要精确复现、故障恢复或 turn-level branching，snapshot 就会变成 environment 的一等公民。至少要区分三类：

| Snapshot                     | 创建位置                           | 用途                                                                   |
| ---------------------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| **Base / reset snapshot**    | task 初始化完成、policy 尚未动作时 | 让 sibling rollouts 共享完全相同的起点                                 |
| **Branch snapshot**          | 某个动作生成之前                   | 从同一个 `(h_t, s_t)` fork 多个 continuation，估计局部反事实 advantage |
| **Sealed terminal snapshot** | episode 结束后                     | 封存后交给独立 verifier，保留最终状态和副作用证据；不再暴露给 agent    |

Snapshot fidelity 的硬性测试不是“restore 没报错”，而是：

```text
restore(S_t); replay(a_t, …, a_n)
    ⇒ 相同 observations、state hashes、terminal reward

fork(S_t, seed=i) 与 fork(S_t, seed=j)
    ⇒ 写入互相隔离，且都不能污染 S_t
```

一个有效 snapshot 至少要覆盖四层状态：

1. **可变世界状态**：filesystem、database、browser cookies/local storage、应用 buffer、tool cache，以及 agent 写入的外部 memory；
2. **熵源**：environment RNG、mock clock、队列顺序和可控服务响应；如果依赖真实互联网或共享 SaaS，就只能声称 partial replay，除非使用 simulator 或 record/replay；
3. **provenance**：task、parent snapshot、environment image、schema/version 和 content hash；
4. **Agent prefix**：world snapshot 并不包含 `h_t`。做 branch 时还必须保存 exact context tokens、retrieval/memory state、behavior-policy version 和 sampling config。

因此真正的 branch point 不是一个裸 `snapshot_id`，而是：

```text
B_t = (snapshot_ref, context_token_ids, memory_ref,
       behavior_policy_version, sampling_config)
```

不同环境的实现方式不同：

| 环境                    | 最小 snapshot backend                                                          | 常见缺口                                              |
| ----------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------- |
| Python state machine    | canonical JSON/MessagePack + content hash；branch 使用 copy-on-write           | 隐藏的 global RNG、mutable singleton                  |
| Stateful API / database | 每 episode 独立 DB/schema + transaction/savepoint 或逻辑 dump                  | 外部 job、cache、共享账户没有一起回滚                 |
| Terminal / container    | immutable base image + 独立 writable overlay；必要时先 quiesce 再做应用级 dump | filesystem layer 不包含 RAM、socket 和未 flush buffer |
| Browser / GUI           | server DB + browser profile + cookies/local storage + fixed clock              | screenshot 或 DOM 不是权威状态                        |
| OS / 长驻进程           | VM/microVM memory snapshot，或退回应用级 checkpoint                            | 恢复成本高，外部设备和网络仍可能漂移                  |

[OSWorld](https://arxiv.org/abs/2404.07972) 为任务提供初始状态配置与 execution-based evaluator，[AppWorld](https://arxiv.org/abs/2407.18901) 用状态单测检查目标和 collateral damage；它们说明为什么“可见页面相同”不足以定义相同状态。后文的 [BPO](https://arxiv.org/abs/2607.14171) 则直接利用中间 snapshot 和 sibling forks 做局部 credit，但它仍是 2026 年的新工作。

### 可以直接使用的 component

| 层                   | 最小实现                                      | 需要扩展时                                                         |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------------ |
| Terminal / code 环境 | Python adapter + Docker + `pytest`/state diff | Terminal-Bench 风格 container tasks                                |
| Stateful API         | 内存数据库 + typed tools                      | [τ-bench](https://arxiv.org/abs/2406.12045) / ToolSandbox 风格环境 |
| Web / GUI            | 本地固定版本网站 + browser driver             | WebArena / OSWorld 风格 evaluator                                  |
| Snapshot substrate   | canonical state + content-addressed manifest  | per-episode DB、copy-on-write overlay、必要时 VM snapshot          |
| Eval harness         | 独立身份 + read-only grader namespace         | [UK AISI Inspect](https://inspect.aisi.org.uk/)                    |
| 高风险执行           | rootless container、无宿主 secrets、默认无网  | gVisor / microVM、episode-scoped credentials                       |

环境最容易被忽略的不是 API，而是 **state authority**。磁盘、应用 buffer 和 rendered GUI 可能同时不同步；“屏幕看起来对”并不等于持久状态正确。每次 mutation 和 completion 最好绑定 state version，并在版本漂移时 fail closed。

---

## 3. Transition Contract：Rollout 必须能重建梯度

下面这个 JSONL schema 比“保存 prompt 和 response”更接近一个可训练的数据契约：

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

五条规则不能省：

1. 保存生成时的 **exact token IDs 和实际 sampler 的 behavior logprobs**；如果做过 temperature/top-p/top-k 变换，最好同时保存 raw-model logprobs。事后重新 tokenize 可能因 chat template、BPE 或 tool-call render 不同而改变序列。
2. 每条 trajectory 固定一个 `behavior_policy_version`。初版系统不要在 episode 中途热更新权重。
3. infra error、timeout 和环境启动失败单独记录，不能静默当成 reward=0。
4. 保存 observation 原文或不可变对象引用，仅有模型“总结后的 observation”不足以重放。
5. branch rollout 必须记录 parent run、branch point 和 snapshot manifest；只有 snapshot、没有相同的 agent prefix，不能算从同一决策状态出发。

还要区分两种“可复现”：给定已记录 actions，**environment replay** 必须恢复相同 observation、state hash 和 reward；重新运行 GPU sampling 得到逐 token 完全相同的轨迹，则可能受 kernel 和 inference engine 影响，不应被当作跨平台硬条件。Invalid action 应消耗一个明确的 step、返回结构化错误且不产生未声明副作用，不能由 harness 悄悄替模型修好。

初始实现可以同步运行：冻结 checkpoint，完成整批 rollout，再做一次 update。等同步闭环正确后再引入异步。否则 throughput 上升的同时，trajectory 可能来自旧 policy，`π_beh` 与当前 `πθ` 的差距会变成隐藏的 off-policy bias。

小规模可用 Hugging Face generation；需要吞吐时用 [vLLM](https://arxiv.org/abs/2309.06180) 或 [SGLang](https://arxiv.org/abs/2312.07104)。但无论使用哪个 server，都要确认 API 能返回真实 token IDs/logprobs，而不是只返回重新编码后的文本。

---

## 4. Verification Contract：Reward 的上限由 verifier 决定

Agent RL 最危险的 bug，是一个错误 artifact 被打成高分。它不只污染一个评测样本，还会产生方向错误的梯度。

Verifier 可以按下面的顺序组合：

```text
权威状态检查 / hidden tests / compiler
                 ↓ hard gate
规则与 policy invariants
                 ↓
独立 semantic judge（只处理剩余开放语义）
                 ↓
高风险样本的人类审计
```

一个实用的 score 不应只返回 `reward: 1`：

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

先保留 reward vector，再决定哪些项进入 objective。Tampering 会让测量本身失效，应隔离并停止使用这条样本；policy violation 和 collateral damage 则进入 hard constraint 或单独的 constraint cost。这样可以画 safety–utility frontier，而不是事后猜测总 reward 为什么变化。

在具有代表性、由人类或更强外部程序裁决的 audit set 上，Verifier 至少要估计两种错误：

- false positive：错误结果被接受，容易把错误行为写进 policy；
- false negative：正确结果被拒绝，主要浪费有效 experience，也可能让策略变得过度保守。

单独追求低 false positive 并不是免费午餐。Hybrid verifier 可能通过更保守来降低误接收，因此要同时报告 FP、FN、分母与置信区间；只有 verifier 输出可解释概率时才报告 ECE/Brier，只有存在多个 verifier 时才报告 disagreement。Deterministic compiler 或 hidden test 不应被硬套概率 calibration。[Reward-model overoptimization](https://arxiv.org/abs/2210.10760) 也说明，持续优化 proxy reward 可能让 proxy 与真实质量逐渐分离。

---

## 5. Update Contract：GRPO 解决了 baseline，不等于解决长程归因

假设同一个 task、同一个初始状态、同一个 behavior `PolicyManifest` 产生 `G` 条 sibling trajectories。最简单的 critic-free estimator 是 RLOO：

```text
A_i = R_i - (1 / (G - 1)) Σ_{j ≠ i} R_j
```

[RLOO](https://aclanthology.org/2024.acl-long.662/) 用其他 samples 的平均 reward 作为第 `i` 条轨迹的 baseline。GRPO 常使用组内标准化：

```text
A_i = (R_i - mean(R₁:G)) / (std(R₁:G) + ε)
```

它们都省去了独立 critic，适合 outcome verifier 强、episode 较短、能为同一 task 采样多个结果的场景。[DeepSeekMath](https://arxiv.org/abs/2402.03300) 是 GRPO 的主要公开来源。

Group 必须由同一 task、同一初始 environment seed 和同一 behavior `PolicyManifest` 构成；不能把难度完全不同的任务混在一起做 reward normalization，再把相对 task 难度误当成 action quality。

如果 siblings 在给定 task/checkpoint 后条件独立、单条轨迹成功概率是 `p`，Bernoulli reward 的整个 group 没有相对信号的概率是：

```text
P(zero variance) = p^G + (1-p)^G
```

任务太容易时全部成功，太难时全部失败；两端都没有 group-relative gradient。这说明 task sampling 确实是 RL estimator 的一部分。但 **有 reward variance 只代表能产生梯度，不代表这批 experience 会改善 held-out capability**。

### Optimization unit：token、turn 和 trajectory 不是同一个目标

如果研究目标是 expected episode return，score-function 项是 `A_i Σ_t log π(a_i,t|h_i,t)`，只能除以与该样本实际长度无关的常数而不改变这个 estimand。下面三个看起来只差一个 `mean` 的目标，实际优化的却是不同的长度加权分布：

```text
L_episode  = (1/B) Σ_i Σ_t ℓ_i,t
L_len_norm = (1/B) Σ_i [(1/T_i) Σ_t ℓ_i,t]
L_token    = (1/Σ_i T_i) Σ_i Σ_t ℓ_i,t
```

`L_episode` 对应 episode-return policy-gradient baseline；`L_len_norm` 让每条 trajectory 的总标量权重近似相同，因此长轨迹的单 token 系数更小；`L_token` 则按本批实际 token 总数形成 ratio estimator。把多轮轨迹拆成 turn samples 后再求均值，还会再次改变每条 episode 的权重。后两者可以是有意降低方差或控制长度的 surrogate，但不能再宣称与原始 expected-return estimand 相同。

GRPO 的 `/std(R_group)` 也不是无害的数值稳定技巧：它会按组内 reward scale 重新加权 task groups，接近零但非零的方差尤其敏感。[Dr.GRPO](https://arxiv.org/abs/2503.20783) 正是从 group-std 与 response-length normalization 的偏差出发修改目标；[STEPO](https://arxiv.org/abs/2607.09773) 则在多轮场景显式约束 turn-level credit mass。后者是 2026 年的新方法，应作为 ablation 候选而不是默认最佳实践。

最小 `LossReducer` 对照是：

```yaml
reduction: episode_sum # vs length_normalized vs token_mean vs turn_mass_conserving
group_reward_scale: none # vs std
```

除最终 success 外，还要按 success/failure、trajectory length、turn count 和 task family 报告每条 episode 的 gradient mass；否则 reward 上升可能只是优化器开始偏爱某一长度区间。

### Dense score 不等于 causal credit

如果同一个 `A_i` 被广播给 trajectory 中所有 action tokens，我们只知道“这条轨迹整体比 siblings 好”，并不知道哪个 tool call 真正造成了成功。一个前面偶然有效的探索动作和后面无关的冗长输出会得到同样符号的 credit。类似地，给每一步一个 dense score 也不自动产生 causal credit：step score 可能只是与最终成功相关，而不是该动作改变了成功概率。

### 什么时候升级 credit estimator

| 观测到的问题                    | 更合适的技术                                                        | 代价/假设                                 |
| ------------------------------- | ------------------------------------------------------------------- | ----------------------------------------- |
| 短任务、强 outcome verifier     | RLOO / GRPO                                                         | trajectory-level credit 较粗              |
| 有可信 step reward              | PPO + critic + [GAE](https://arxiv.org/abs/1506.02438)              | critic 计算与偏差                         |
| 只需在 turns 间重分配总 credit  | STEPO 式 mass redistribution                                        | 改变权重，不识别反事实因果                |
| 同一状态在轨迹中重复出现        | [GiGPO](https://arxiv.org/abs/2505.10978) 式 anchor-state grouping  | 需要可靠 state matching                   |
| 有冻结 reference 和已知目标答案 | [TRACE](https://arxiv.org/abs/2607.13988) 式 tool-boundary TD proxy | proxy quality 依赖 reference；2026 新证据 |
| 环境可以 snapshot/restore       | [BPO](https://arxiv.org/abs/2607.14171) 式 sibling branches         | fork 成本与 restore fidelity；2026 新证据 |
| 轨迹跨越 context window         | segment critic / compaction + cross-segment credit                  | summary 可能丢失充分状态                  |

对于 snapshotable 环境，可以在同一个状态 `s_t` fork `K` 个动作并 rollout 到结束：

```text
S_t = env.snapshot();  H_t = exact_agent_prefix()

for k in 1..K:
    env_k = env.fork(S_t, branch_id=str(k))   # 继承相同 env RNG state
    τ_t,k = rollout(env_k, prefix=H_t, policy=π_beh,
                    policy_sampling_seed=k)

A_t,k = G_t,k - (1 / (K - 1)) Σ_{j ≠ k} G_t,j
```

因为 sibling continuations 共享同一个世界状态和 agent prefix，这个比较比从不同初始轨迹减均值更接近局部反事实。`branch_id` 只负责隔离写入，不能偷偷改变 environment randomness；探索差异来自 policy sampling。实际实现还必须给每个 branch 相同的 remaining-step/token budget，并验证 branch isolation；否则“更高 return”可能只是某一支获得了更多计算。它也只在 snapshot 真实、环境随机性被控制时成立，不是任意开放环境中的万能因果识别。

无论选择哪一种 `CreditEngine`，都应增加一个小规模因果审计：在固定 snapshot 上删除、替换或重新采样被赋予高 credit 的 action，测量 terminal return delta，并报告 estimated credit 与 intervention delta 的 rank correlation。它不能证明全局最优，但能排除“漂亮的 dense score 与真实作用完全无关”。同时始终保留 outcome-only arm，避免 process reward 自己成为新的可攻击 proxy。

一个重要区分是：DAPO/DPPO 主要改善 sampling、loss aggregation 或 trust region；它们不会自动回答“哪个 turn 导致成功”。[DAPO](https://arxiv.org/abs/2503.14476) 的 dynamic sampling 可以减少零方差 group，[DPPO](https://arxiv.org/abs/2602.04879)（2026 preprint）用更直接的 KL/TV divergence 估计约束更新；credit assignment 仍需单独设计。

---

## 6. Update Contract：把 estimator、objective 和系统分开

### 先写清楚三个 policy

一个可靠的实现不能把所有旧模型都叫作 `old_policy`：

| 符号     | 角色                                     | 必须保存什么                                                   |
| -------- | ---------------------------------------- | -------------------------------------------------------------- |
| `π_beh`  | 真正产生 action 的 behavior distribution | generator build、完整 sampler/grammar、逐 action-token logprob |
| `π_prox` | clipping / trust-region 的中心           | checkpoint version，以及它是否等于 `π_beh`                     |
| `π_ref`  | 冻结的能力锚点                           | reference revision、KL 方向与估计方法                          |

同步单步训练里 `π_beh = π_prox` 很常见；异步、replay 或 generator/trainer 数值不一致时则不能默认相等。如果要把 behavior data 纠正到 proximal policy，再围绕 proximal policy 更新，概念上对应：

```text
w_off  = π_prox(a|h) / π_beh(a|h)
r_prox = π_θ(a|h)    / π_prox(a|h)

π_θ(a|h) / π_beh(a|h) = r_prox · w_off
```

具体 estimator 可以截断、拒绝或合并这些 ratios，但必须说明改的是哪一个量。[PPO](https://arxiv.org/abs/1707.06347) 只在其数据与旧 policy 语义成立时提供 proximal update；[AReaL](https://arxiv.org/abs/2505.24298) 则明确处理 asynchronous behavior 与 proximal policy 的分离。

`π_ref` 又是另一件事：它不是 importance-sampling 分母。`log πθ - log πref` 只有在声明的采样分布下，才对应目标 KL 的 Monte Carlo estimator；在 stale behavior samples 上直接取均值，不能自动叫作当前 policy 的 forward KL。实验必须注明 KL 的方向、采样分布、是否使用全词表审计或 non-negative estimator，以及它只进入 reward、只进入 loss，还是被重复计算了两次。

### Reward、ratio 和 clipping 的粒度也要对齐

Agent 常得到 sequence-level outcome reward，但必须区分真正的 change-of-measure ratio 和为稳定性设计的 length-normalized score：

```text
token IS ratio:       r_t = exp(log πθ(a_t|h_t) - log πold(a_t|h_t))
trajectory IS ratio:  R_IS = exp[Σ_t log r_t]

turn normalized score:     s_turn = exp[(1/T_turn) Σ_t log r_t]
episode normalized score:  s_seq  = exp[(1/T_episode) Σ_t log r_t]
```

`R_IS` 才是真正的 trajectory importance ratio，但方差会随 horizon 爆炸；`s_turn/s_seq` 是有意改变目标的几何均值 surrogate，不能拿来声称完成了无偏 trajectory off-policy correction。[DAPO](https://arxiv.org/abs/2503.14476) 使用 token-level policy loss 与 asymmetric clipping；[GSPO](https://arxiv.org/abs/2507.18071) 使用 length-normalized sequence likelihood-ratio score 和 sequence-level clipping，尤其关注 MoE 训练稳定性。两者不是可以在实现里静默互换的“小技巧”。多轮 Agent 可把 turn-normalized score 作为第三个 ablation，并分别报告 token、whole-turn 和 whole-episode clip fraction；只有系统真的丢弃样本时才另报 rejection fraction。

如果每批 rollout 都来自当前完整 `PolicyManifest`，`π_beh = π_prox = πθ`（在梯度计算开始时），generator/trainer logprob 也通过一致性 gate，而且只做一次 optimizer step，那么 expected episode return 的最小目标可以写成 RLOO/REINFORCE 加 KL：

```text
L_pg = -(1/B) Σ_i A_i · [Σ_{k∈M_i} log πθ(y_i,k | y_i,<k)]

L = L_pg + β KL(πθ || πref)
```

它无 critic、无 replay，也没有必要假装 ratio clipping 正在解决问题。这里若再除以样本自己的 `|M_i|`，就已经切换成上一节的 length-normalized surrogate。

只要 `πθ ≠ π_beh`——无论来自同一 batch 的多 epochs、async/replay staleness，还是需要校正的 generator/trainer mismatch——就必须显式选择 importance correction、clipping、rejection 或 quarantine。下面的普通 PPO-style clipped surrogate只适用于 `π_prox = π_beh` 的特殊情形；三者不同时应回到前述 three-policy contract：

```text
ρ_t(θ) = exp(log πθ(a_t|h_t) - log πbeh(a_t|h_t))

L = -E[min(ρ_t A_t,
           clip(ρ_t, 1-ε, 1+ε) A_t)] + β KL(πθ || πref)
```

[PPO](https://arxiv.org/abs/1707.06347) 的 ratio clip 控制 sampled action 上的变化；`clip_fraction`、KL、entropy、gradient norm 和 importance-ratio tails 必须一起监控。

这里的 reduction 直接复用上一节预注册的 `LossReducer`；不能在训练框架默认值里悄悄决定 token、turn 或 trajectory 权重。

最小训练循环不需要复杂：

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

这里故意把三个东西分开：

- `advantages` 决定 credit estimator；
- `reinforce_with_kl_loss` 决定 update objective；如果复用 batch，可替换为 PPO-style clipped objective；
- `collector/registry` 决定数据是否 near-on-policy。

这样替换 RLOO、GAE 或 branching credit 时，不需要重写环境和 verifier。

### Average entropy 会掩盖真正的 exploration collapse

在数学 reasoning RLVR 中，近期工作分别从 gradient covariance 和 high-entropy minority tokens 观察到：少量 token 可能承载不成比例的 exploration signal。[Entropy Mechanism](https://arxiv.org/abs/2505.22617)、[High-Entropy Minority Tokens](https://arxiv.org/abs/2506.01939) 对 Agent 的待检验假设是，这些 token 是否真的对应 tool choice 或行为分叉；不能从 reasoning 证据直接假定映射成立。

`ExplorationMonitor` 至少应按 advantage sign、entropy quantile、turn index、tool/action type 和 success/failure 分层记录 entropy 与 gradient mass，并检验高-entropy token 是否与实际 branch/action intervention 对齐；再把 held-out `pass@k`、tool-sequence diversity 与 `pass@1` 一起画。这样才能区分 reward 导致的合理收敛、clipping 自己造成的分布偏置，以及关键决策点的过早坍缩。

### Generator–trainer async：吞吐优化会改变数据分布

这里要把两个经常都被叫作 `actor` 的角色分开：

- **Generator / rollout worker** 持有只用于推理的 behavior policy `π_beh^v`，与环境交互并写 immutable trajectories；
- **Trainer / learner** 持有 master parameters、optimizer state 和当前训练版本 `π_train^u`，消费轨迹并发布新权重。

完整数据流是：

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

定义 version lag：

```text
Δ_version = u - v
Δ_time    = trainer_consume_time - rollout_finish_time
D_sample  = (1/|M|) Σ_{t∈M(τ~π_beh^v)}
            |log π_train^u(a_t|h_t) - log π_beh^v(a_t|h_t)|
```

`Δ_version` 只是系统年龄；`D_sample` 也只是 **在 behavior-sampled action mask 上** 的 drift diagnostic，不是完整 policy-space distance。相同的一版延迟，在不同 learning rate、KL、update size 和采样分布下可能对应完全不同的 `D_sample`。因此版本 gate 与 importance-ratio/drift gate 都要有。

同步与异步不是一个布尔开关，而是至少三种不同的数据语义：

| 模式                          | Generator 与 trainer 的关系                                   | 数据语义                                                               |
| ----------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Synchronous barrier**       | `π^v` 完成整批 group，trainer 才更新为 `π^(v+1)`              | `Δ_version=0`；最干净，但受最长 trajectory 拖累                        |
| **Pipelined / bounded async** | generator 连续运行；trainer 只接收 `Δ_version≤L` 的完整 group | bounded-staleness；只有同时通过 drift/ratio gate 才能称 near-on-policy |
| **Unbounded async / replay**  | trainer 可反复使用任意旧版本数据                              | 真正 off-policy；不能再沿用未经修改的 on-policy recipe                 |

Reference implementation 应先固定 `L=0`，跑通后只把 `L=1` 作为单独 ablation；不要一开始就用“队列里有什么训什么”。对 RLOO/GRPO，**同一个 group 内的所有 siblings 还必须来自同一个 behavior version**。混合 `π^v` 与 `π^(v+1)` 后，组内差异同时包含 action quality 和 policy drift。

Async 的调度单元应该是一个完整 group lease，而不是单条先到先训的 trajectory：

```text
lease = (group_id, behavior_manifest_hash,
         base_snapshot_id, sampler_config_hash)
```

所有 siblings 共享 lease；advantage 只能在 group 完整并验证后计算。某个 sibling timeout 时，要按预注册规则丢弃/降级整个 group，不能用后来生成的新 policy 样本补位。

#### In-flight episode 到底怎么办？

长程 Agent rollout 很可能在 trainer 发布新版本时仍未结束。只有三种语义清楚的选择：

1. **finish-old**：episode 继续由旧 generator 完成；新 task 路由到新版本。这最干净，但需要双 buffer 或暂时保留旧 worker；
2. **abort-and-restart**：丢弃未完成轨迹，从原始 snapshot 用新版本重启；语义清楚，但浪费 rollout；
3. **mixed-policy continuation**：中途换权重继续生成。严格 on-policy 基线应禁止；若一定要做，必须保存每个 action/token 的 behavior version 和 logprob，并使用支持这种数据的 off-policy estimator。

`pause → load new weights → resume` 是 serving capability，不自动保证 RL 正确。如果请求在 pause 前后使用了不同参数，它就是 mixed-policy trajectory；旧 KV cache 也不能直接与新权重混用。[vLLM 的原生 RL API](https://vllm.ai/blog/2026-05-28-native-rl-apis) 提供 weight-transfer 与 `abort`/`wait`/`keep` pause 模式，但算法层仍要明确选择上述语义。

[DORA](https://arxiv.org/abs/2604.26256) 是另一种 2026 新方案：同时保留多个 rollout policy version，让长 trajectory 在原版本上结束，并以 bounded staleness 控制 trainer admission。它很适合作为 multi-version serving 的设计参考，但仍是新预印本证据。

#### Partial rollout、截断和 compaction 是数据语义

“episode 没有完整结束”可能表示多种统计语义不同的事件：

```text
agent_finish | env_terminal_success | env_terminal_failure
task_budget_exhausted_terminal
max_step_truncation | max_token_truncation
tool_timeout | infra_abort | scheduler_cancel
```

前三类表示 policy 主动结束或 environment 给出了 terminal outcome，但仍要交给 verifier 判断成功与否。预先写入 task contract 的 finite-horizon budget exhaustion 也可以合法定义为 terminal failure；相反，harness 临时截断、tool timeout、infra abort 和 scheduler cancellation 属于 censored/system events。关键不是枚举名称，而是预算是否在任务定义中事先固定。把后一类事件静默写成 reward=0，会把 censored trajectory 当作失败，并系统性改变对长任务的训练权重。

[APRIL](https://arxiv.org/abs/2509.18521) 通过 over-provision requests、先收集目标数量的已完成样本，再把未完成 prefix 带到后续 iteration 来减少 rollout 长尾；[RollPacker](https://arxiv.org/abs/2509.21009) 提供了尽量保持同步语义的另一类对照。由 first-completion admission 推断，它**可能**形成偏向快/短轨迹的 temporal curriculum，但这是需要按 length、difficulty 和 reward 分层验证的风险，不是论文已经证明的结论。

要继续一个 partial rollout，`TrajectorySegmentStore` 必须保存：

```text
segment_id, continuation_parent,
snapshot_id, exact_context_token_ids, memory_state_ref,
behavior_policy_version, behavior_logprobs, sampler_config,
remaining_step/token_budget, termination_reason
```

然后预注册四个对照：旧版本 pinned continuation、mixed-version continuation、abort-and-restart、同步等待完整轨迹。GRPO/RLOO 的一个 group 还必须在 continuation 后保持相同语义；不能拿后来新 policy 生成的 sibling 给缺失位置补位。

Context compaction 同样属于 policy boundary，而不是无损的日志压缩。summary 决定后续 policy 看见什么，也可能丢掉 verifier 所需状态。[CompactionRL](https://arxiv.org/abs/2607.05378) 将 summary generation 与跨 segment credit 一起优化，是这一方向的 2026 新证据。最小实现应记录 compaction 前后 token hashes、compactor model/version、summary token IDs，以及 compaction 发生在哪个 snapshot；评测时保留 no-compaction control，并单独测 summary omission 与 restore fidelity。

#### 权重发布必须是原子的

不要让 generator 在部分 tensor 或部分 TP/PP/EP ranks 已更新时接收新请求。一个最小 publish protocol 是：

```text
trainer 完成 u+1
→ 写 immutable manifest（model/tokenizer/template/checksum）
→ 将权重 stage 到所有 inference ranks 的 shadow slot
→ 全部 ranks 校验 checksum 并 ACK
→ 在声明的 policy-lease boundary 原子切换 active_version
→ 若同一逻辑 episode 继续，invalidate KV cache 并用新权重重新 prefill
→ 才允许 scheduler 将后续请求路由到 u+1
```

Pinned-policy 基线的 lease boundary 是 episode boundary；只有显式 mixed-policy 实验才允许在记录完整的 segment/turn boundary 切换，并把后续段标成新 behavior version。不能在任意 token 位置静默热切换。

同卡、分时部署可以用训练/推理 resharding；独立 GPU pool 则需要 NCCL、CUDA IPC、RDMA 或 checkpoint/object-store transport。[HybridFlow/verl](https://arxiv.org/abs/2409.19256) 的重点正是把 RL dataflow 与模型放置/resharding 分开。若只训练 LoRA，可以只同步 adapter，但 `base_revision + adapter_revision` 仍然必须形成一个不可分割的 policy version。

另一个隐蔽问题是 trainer 用 BF16，而 generator 用 FP8/量化推理。即使 version ID、权重 checksum 都相同，两边也可能因为 kernel、并行布局、MoE routing、constrained decoding 或 sampler implementation 算出不同 logprob。这是 **training–inference mismatch**，和“数据来自旧版本”的 async staleness 是两个独立误差源：

```text
δ_TIM,t   = log π_generator^v(a_t|h_t) - log π_trainer^v(a_t|h_t)
δ_stale,t = log π_trainer^v(a_t|h_t)   - log π_trainer^u(a_t|h_t), u > v
δ_total,t = log π_generator^v(a_t|h_t) - log π_trainer^u(a_t|h_t)

δ_total,t = δ_TIM,t + δ_stale,t
```

纯 TIM 必须在同版本 `v` 上比较 generator 与 trainer；纯 staleness 则要在同一 trainer backend 上加载 `v` 和 `u` 比较。`generator-v` 对 `trainer-u` 只能叫 total gap。先保留逐 token signed delta，再分别汇总均值、`p95/max |δ|`；不能从两个绝对值相减声称完成误差分解。

在 frozen checkpoint 上，`LogprobConsistencyAuditor` 应固定 exact prefixes 和 sampled token IDs，让两边逐 token 重算概率，并报告上述 TIM metrics、sequence log-perplexity difference 和 non-negative `k3` estimator。容差应先从当前硬件和纯 BF16 同引擎 baseline 预注册，而不是看到结果后选择。[Diagnosing Training–Inference Mismatch](https://arxiv.org/abs/2605.14220) 在隔离 policy drift 后仍观察到小数值差异可独立破坏训练；[verl 的 rollout-correction 实现](https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md) 提供了相应 metrics、importance correction 与 rejection hooks。两者都是当前实现依据，不代表 correction 可以替代 zero-mismatch diagnostic。

Tokenizer、chat template、tool rendering、grammar、quantization、routing replay 和 inference engine 都必须进入 `PolicyManifest`；否则 ratio drift 可能来自实现不一致，而不是 policy update。

#### Stale data 怎么处理？

PPO clipping 只能限制已有 sample 上的 update，不能把严重过期的数据神奇地变回 on-policy。长轨迹上直接连乘 importance ratios 又会产生极高方差。实际顺序应该是：

这里的 `behavior_logprobs` 必须对应**实际 sampler distribution**。如果 generator 使用 temperature、top-p 或 top-k，却只保存未经这些变换的 raw-model logprob，importance ratio 的分母就是错的；截断采样还可能破坏 off-policy correction 所需的 support。最小基线最好使用 temperature=1、无 top-k/top-p truncation，或完整保存变换后的采样概率。

1. 先用 version-lag cutoff 和 group-homogeneous batching 限制 off-policyness；
2. 保存 rollout-time behavior logprobs，在 trainer 端重算 current logprobs，监控 ratio p50/p95/p99 与 effective sample size；
3. 对轻度 drift 使用 ratio clipping、KL gate 或 sample rejection；
4. 只有已有 turn-level critic 时，再考虑 [IMPALA/V-trace](https://arxiv.org/abs/1802.01561) 一类截断 importance correction；
5. 超出预注册 lag/ratio gate 的 trajectory 丢弃或只用于离线分析，不要静默训练。

异步系统还是一个 rate-control 问题。如果 generator 产出率 `λ_gen` 长期高于 trainer 接收率 `λ_train`，queue 会增长，数据必然越来越 stale。需要 queue age/high-watermark、generator backpressure、动态 worker allocation 或 stale-sample quarantine；不能靠无限 buffer 掩盖失衡。还应按 task family、trajectory length 和 reward 分层报告 stale/drop rate——否则系统可能优先丢掉长而难的样本，暗中改变 curriculum。

[AReaL](https://arxiv.org/abs/2505.24298) 展示了 generation/training 完全解耦、受控 staleness 与 staleness-aware PPO；[Asynchronous RLHF](https://arxiv.org/abs/2410.18252) 则直接研究了异步带来的 off-policy 性能权衡。但两者的主要证据来自 reasoning/instruction-following，不应被直接写成多轮工具 Agent 的已解决结论。

最小 controller 可以这样写：

```python
MAX_VERSION_LAG = 0  # sync baseline; later ablate with 1

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

每个 group 还需要 immutable `group_id` 和 exactly-once consumption。Trainer checkpoint 应同时提交 `consumed_group_ids + optimizer_step + output_policy_version`；进程恢复后既不能重复应用同一 group，也不能把已经更新但尚未 ACK 的 group 当作未消费。

Async 是否值得，不看 tokens/s 一项。至少同时比较同步基线与异步版本的 held-out success、安全、KL、ratio tails、stale-drop rate、queue age、generator/trainer utilization、weight-sync latency 和 wall-clock-to-target。吞吐提高但单位时间学到的 held-out capability 下降，不算成功。

### Component 选择

| 拓扑                 | Generator                        | Trainer / weight sync                                                           | 适合目标                               |
| -------------------- | -------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------- |
| 单卡同步 smoke       | 同进程 generation                | PyTorch/PEFT，函数调用直接复制 LoRA                                             | 验证 trace、reward、梯度闭环           |
| 多卡 colocated sync  | vLLM rollout phase               | [verl / HybridFlow](https://arxiv.org/abs/2409.19256) reshard 到训练 phase      | PPO/GRPO/RLOO 可复现基线               |
| 分离式 bounded async | vLLM/SGLang generator pool       | FSDP/DeepSpeed learner + versioned queue + NCCL/IPC/RDMA sync                   | 长轨迹、减少 straggler idle            |
| 全异步研究系统       | continuous generator pool        | [AReaL](https://arxiv.org/abs/2505.24298) 风格 learner/staleness controller     | 明确研究 off-policy trade-off          |
| 多版本 streaming     | 同时保留若干 pinned policy pools | [DORA](https://arxiv.org/abs/2604.26256) 风格 lease/routing + bounded admission | 让长 episode 在原版本完成；2026 新证据 |
| 既有 agent runtime   | trace adapter                    | [Agent Lightning](https://arxiv.org/abs/2508.03680) 风格 disaggregation         | 将执行与训练解耦                       |

可以直接阅读的实现锚点也很明确：[slime `train_async.py`](https://github.com/THUDM/slime/blob/main/train_async.py) 在更新权重前等待当前 generation，适合先理解半异步；[verl fully-async recipe](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md) 显式拆成 Rollouter、MessageQueue、Trainer 和 ParameterSynchronizer；[THUDM AgentRL](https://github.com/THUDM/AgentRL) 给出了 Ray 上多轮 Agent 的 rollout/actor/reference worker pools 与 `group_id` queue。这些代码能提供 component，但不能替代前面的 version、logprob 和 held-out gates。

最小 repo 不需要超过这些模块：

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
├── eval.py           # held-out + retain + integrity suites
└── tests/           # replay, masks, versions, corruption, logprob contract
```

先同步，再把 bounded async 当作一次算法与系统共同变化的 ablation；吞吐不是免费的算法改进。

---

## 7. 一个真正可以开始跑的 smoke experiment

下面不是 paper-scale recipe，而是用来证明闭环没有作弊的最小实验。

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
  branching: false # 先证明 restore fidelity，再开启 turn-level forks

rollout:
  mode: synchronous
  group_size: 8
  temperature: 1.0
  top_p: 1.0
  top_k: null # 禁用 truncation；否则保存真实 sampler logprob
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

任务可以从确定性的文件、JSON、schema、CLI 修复开始。每个 task 包含：初始 container snapshot、自然语言目标、允许的 tools、隐藏 grader 和期望保持不变的 invariants。Train/held-out 应按 task template 或 environment family 隔离，而不是只换几个数字。

如果基础模型在 smoke set 上连 typed action 都无法稳定生成，先补少量 tool-call / recovery SFT；不要期待 RL 从全失败、全 parse error 的分布里凭空产生有效探索。

### 四个对照

1. frozen no-update baseline；
2. 相同 cold-start 的正常 RLOO update；
3. reward 在 group 内随机打乱的 null update；
4. SFT-only control，训练 token 数与 RL update 尽量匹配。

### Go / no-go gate

在看 capability gain 前，先要求：

- reset 后的 state hash 可复现；
- restore 后重放固定 action suffix，observation、state hash 和 reward 100% 一致；
- 两个 branch 的 mutation 彼此隔离；sealed terminal snapshot 不可变且不再对 agent 可见；
- rollout、trainer 和 evaluator 引用同一个完整 `PolicyManifest`；scaffold drift 为 0；
- 每条 action token 都有长度对齐、有限的 behavior logprob；
- `π_beh`、`π_prox`、`π_ref` 的版本、用途和 logprob 字段无歧义；
- 每个 episode 只有一个 behavior-policy version；
- 每个 sibling group 只有一个 behavior-policy version，mixed-policy episode 数为 0；
- generator 所有 ranks 的 active-weight checksum 与 registry 一致；
- frozen checkpoint 下 generator/trainer 的 `p95/max |Δlogp|` 与 `k3` divergence 在预注册容差内；
- queue 中不存在超过 `max_version_lag` 却仍进入 trainer 的样本；
- group ID exactly-once consumption；duplicate/lost/incomplete group 分开计数；
- `max_version_lag=0` 的 async code path 与同步实现产生数值一致的 loss/gradient；
- stale/drop rate 按 task family、trajectory length 和 reward 分层，不隐藏 curriculum bias；
- sample index、task ID、seed 和 checkpoint 完整；
- `finish`、environment terminal、truncation、infra error 与 scheduler cancellation 使用不同枚举；censored trajectory 不被静默写成 reward=0；
- `LossReducer` 的 unit test 证明 `episode_sum` 实现了长度无关归一常数下的 score-function 项；其他 reduction 被显式标为不同 surrogate；
- verifier 在具有足够分母、由人类或更强程序裁决的代表性 fixtures 上通过预注册的 FP/FN gate；
- 在独立 toy-logit unit test 中，正/负 advantage 的 aggregate policy-gradient directional derivative 符号正确；真实共享参数 batch 只要求总体目标按预期下降且 KL 有界，不要求每个 sampled action 单调变化；
- shuffled-reward control 不出现稳定 held-out gain。

最后同时报告：held-out success、retain-suite delta、`pass^3`、invalid-action rate、tool calls/token/latency、side-effect rate、verifier audit-set FP/FN（含分母与 CI）、termination-reason distribution、zero-advantage group ratio、按长度分层的 gradient mass、KL、gradient norm 和 bootstrap confidence interval。异步时再报告 version/time lag、queue age、stale-drop rate、weight-sync latency、generator/trainer utilization 与 training–inference mismatch metrics；使用 clipped objective 时报告相应粒度的 clip fraction，只有显式丢弃样本时才另报 rejection fraction，并始终给出 importance-ratio tails。

在 policy、scaffold、budget 都固定，且 attempts 是独立或至少 exchangeable draws 的前提下，`pass@3` 表示三次里至少成功一次，更像搜索能力；`pass^3` 表示三次全部成功，更接近生产可靠性。这个区分来自 [τ-bench](https://arxiv.org/abs/2406.12045)。但每题只有三次的估计方差很高；生产可靠性结论仍需更多重复、per-task uncertainty 和置信区间。

---

## 8. 从“能跑”到“有研究价值”

闭环跑通以后，有五条比继续换 optimizer 更值得研究的方向。

### 8.1 哪些 experience 真正形成能力？

中等 pass rate 或非零 group variance 只说明当前 task 有学习信号，不说明训练它会改善其他任务。

更强的 estimand 是：

```text
u(g, s) = J_heldout(Update(θ₀, rollout(g, s))) - J_heldout(θ₀)
```

所有 candidate update 都从同一个 checkpoint `θ₀` 独立开始，定义的是 noisy one-update effect estimand，而不是自动识别出的因果效应。还需要 paired evaluation seeds、no-update/null control、多个 rollout/update seeds、置信区间，以及从多个候选中挑最大结果时的 selection correction，才能把变化归因于该 experience。这也是我们在 CUES-TMax 中试图测量的问题；在正式 causal updates 完成前，不能把 behavior taxonomy 或 surrogate 当作 utility label。

因此 `TaskSampler` 不应只有“保留 non-zero-variance groups”一个按钮。一个可审计 curriculum 至少维护四条流：当前可学习的 tasks、只有少量正例的 hard tasks、已经掌握但需要防遗忘的 retain tasks，以及环境本身不可行或 verifier 不可信的 quarantine tasks。每个 sample 还要保存 sampling propensity 和 capability family，才能判断训练分布是否在不知不觉中缩窄。[TMax](https://arxiv.org/abs/2606.23321) 是展示 difficulty control、persona 和 verifier diversification 的 2026 新预印本；这里更进一步要求用独立 one-update intervention 区分 learnability 与 causal utility。这是我在 CUES-TMax research protocol 中预注册、但尚未完成正式 causal updates 的问题。

### 8.2 Target gain 是否以通用能力退化为代价？

当前 RL prompts 上的 reference KL 只约束被采样上下文附近的输出，不能保证未采样领域、旧工具 schema、instruction following 或 safety refusal 不退化。`RetainEvaluator` 应冻结一套 task-family-disjoint probes，每隔固定 updates 同时测 target gain 与 retain loss；发生越过预注册 frontier 的退化时，停止或回滚 checkpoint。

最小对照是 fixed KL、SFT/PTX replay、domain-balanced replay 和 no-retention control，并画 `target gain – retain loss` Pareto，而不是只报训练 KL。[InstructGPT](https://arxiv.org/abs/2203.02155) 已使用 pretraining-data mix 缓解部分能力回退；[RECAP](https://arxiv.org/abs/2510.21978) 更直接讨论 current-task KL 无法保护 broader capabilities，但证据来自视觉语言模型且仍是新预印本，应把它当作 Agent 实验动机而非已完成外推。

### 8.3 Verifier error 会不会随 horizon 累积？

待检验假设是：horizon 增长会增加遇到冲突文档、stale state、wrong-path artifact、semantic lure 和模糊 rubric 的机会。实验应固定 experiment-owned token budget、注入可控 corruption，并分别画 actor error、verifier FP/FN、calibration、reward-hack rate 与 horizon 的关系；但固定 token 数只能减少 length confounding，不能消除 task difficulty、content density 和 tool topology 等混杂。不能只看一个 LLM judge 的平均正确率。

我们当前的 verifier-horizon pilot 观察到了 horizon effect，但没有达到预注册的 superlinearity 标准，且每个 cell 只有 14 episodes，低于预注册的 50。V3 小样本里 hybrid verifier 的 FP 是 `0/7`，代价是 FN 达到 `5/9 (55.6%)`，高于 semantic judge 的 `3/9 (33.3%)`。因此它支持的是继续检验的协议，而不是“hybrid 免费更好”或“误差必然超线性增长”的结论。完整实验设计、失败 gate 和结果见[三阶段 verifier-horizon 实验](https://ajing.github.io/posts/2026-08-14-verifier-error-horizon-scaling/)。

Verifier assets 还必须由独立身份 provision 到 agent 无写权限的 read-only namespace：agent 不可修改 tests、scorer、clock 或 reference artifact。[METR 的真实 agent traces](https://metr.org/blog/2025-06-05-recent-reward-hacking/)记录了修改 scorer/tests、利用 metadata 等 reward hacking；[OpenAI 的 CoT monitoring 研究](https://openai.com/index/chain-of-thought-monitoring/)说明当前 monitor 可以发现部分 hack，但直接把 monitor signal 作为优化目标也可能让意图更难观察。因此 monitor 更适合做 quarantine/audit signal，而不是未经验证地并入 dense reward。

### 8.4 Agent 是否真的完成了状态改变？

工具调用成功、GUI 显示成功和权威持久状态正确是三件事。对 GUI/CLI hybrid agent，mutation 应绑定 generation fence，写入权限应是单次、原子、可审计的；无法证明状态一致时，abstention 比 false completion 更正确。这个设计边界来自我近期 ReplicaGuard prototype 的状态模型：disk 是 authority，dirty buffer 是受保护意图，GUI 只是证据；当前结果仍来自受控 loopback 环境，不能外推成真实世界 prevalence。

### 8.5 Benchmark 与 user simulator 也可能被 policy 学会利用

Agent eval 应拆成五条轴：capability、重复运行可靠性、安全/副作用、成本/时延和 eval integrity，不能由一个 success rate 或 LLM judge 总分覆盖。`EvalFirewall` 至少使用 temporal/private holdout，分别审计 template、environment、repository 的 exact/semantic overlap；去掉 task identifiers 只能减少显式识别，不能阻止 semantic fingerprinting。网络和答案来源控制还必须与被测 construct 相容：对 open-web research agent，封网会改变任务本身，应公开 allow/block policy 及其 realism 代价。最终 artifact 由独立身份在 read-only grader namespace 中复验，并公开 system prompt、tools、预算、retries 和 scaffold manifest。这些措施降低而不消除 contamination。

这不是假想问题。OpenAI 审计的是 o3 在 64 次运行中未稳定解决的 138 题——即榜单的 27.6% 困难子集，而不是随机样本；其中 59.4% 有 material test/description issues。加上 frontier models 能复现部分 gold patch/题目细节的 contamination 证据，OpenAI 因这两个原因停止报告该榜单。[OpenAI SWE-bench audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)

Anthropic 在 Claude Opus 4.6 的 multi-agent BrowseComp 运行中检查 1,266 题：9 例从公开 benchmark materials 获得答案，另有 2 例先识别自己在评测、再定位并解密 answer key。作者同时强调任务没有禁止搜索这些来源，因此不把行为本身定性为 alignment failure；它证明的是 web-enabled static eval 的完整性风险。[Anthropic BrowseComp eval awareness](https://www.anthropic.com/engineering/eval-awareness-browsecomp)

多轮客服或协作任务里，user simulator 本身还是另一个 policy，属于 transition dynamics。`SimulatorMatrix` 应版本化 user model、prompt、tools 和 sampling config，跨多个 simulator families 评测，并留一个 held-out simulator 或真实对话审计集。[τ²-bench](https://arxiv.org/abs/2506.07982) 将 agent 与用户双方都能操作环境建模为 dual-control Dec-POMDP；[RealUserSim](https://arxiv.org/abs/2605.20204) 是 2026 新预印本，其偏离与改善结论限定在 WildChat-derived profiles、五个行为维度和 τ-bench 实验，不能外推成所有 simulator 与真实用户场景的既定事实。

这五条方向分别对应 experience utility、能力保留、reward integrity、runtime correctness 和 eval integrity。它们共同说明：Agent RL 的研究对象不是一条 loss，而是整个闭环中“哪一个信号可以被相信”。

---

## 结语

如果你只复现本文的一件事，就复现下面这条链：

```text
同一个完整 PolicyManifest
→ 同一个可重置初始状态
→ 一组 fresh sibling rollouts
→ agent 无写权限的独立 verifier namespace
→ 明确的 advantage estimator
→ 恰好一次可审计 update
→ task-disjoint held-out evaluation
```

这条链跑通以后，PPO、GRPO、DPPO、branching credit 或异步 rollout 都只是可以替换的 component。跑不通时，继续增加数据、模型和 GPU，只会让错误更昂贵。

Agent RL 的技术深度，最终不在于用了多少算法名，而在于能否回答：**这个 experience 为什么产生了这个梯度，而这个梯度为什么改善了一个未见过的任务？**

---

## 精选一手参考

- [POMDP: Planning and Acting in Partially Observable Stochastic Domains](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [RLOO: Back to Basics](https://aclanthology.org/2024.acl-long.662/)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [STEPO / EvoCUA-1.5](https://arxiv.org/abs/2607.09773)（2026 preprint）
- [Rethinking the Trust Region / DPPO](https://arxiv.org/abs/2602.04879)（2026 preprint）
- [GiGPO](https://arxiv.org/abs/2505.10978)
- [TRACE](https://arxiv.org/abs/2607.13988)（2026 preprint）
- [Branching Policy Optimization](https://arxiv.org/abs/2607.14171)（2026 preprint）
- [Entropy Mechanism of RL](https://arxiv.org/abs/2505.22617)
- [High-Entropy Minority Tokens](https://arxiv.org/abs/2506.01939)
- [TMax: A Simple Recipe for Terminal Agents](https://arxiv.org/abs/2606.23321)（2026 preprint + open code/data）
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [RECAP: Mitigating General Capability Regression](https://arxiv.org/abs/2510.21978)（新预印本；VLM evidence）
- [Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Agent Lightning](https://arxiv.org/abs/2508.03680)（2025 preprint）
- [Tool-calling Pipeline Sensitivity](https://arxiv.org/abs/2606.00135)（ICML 2026）
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256)
- [IMPALA / V-trace](https://arxiv.org/abs/1802.01561)
- [Asynchronous RLHF](https://arxiv.org/abs/2410.18252)
- [AReaL](https://arxiv.org/abs/2505.24298)
- [DORA](https://arxiv.org/abs/2604.26256)（2026 preprint）
- [Diagnosing Training–Inference Mismatch](https://arxiv.org/abs/2605.14220)（2026 preprint）
- [APRIL: Active Partial Rollouts](https://arxiv.org/abs/2509.18521)
- [RollPacker](https://arxiv.org/abs/2509.21009)
- [CompactionRL](https://arxiv.org/abs/2607.05378)（2026 preprint）
- [vLLM Native RL APIs](https://vllm.ai/blog/2026-05-28-native-rl-apis)
- [τ-bench](https://arxiv.org/abs/2406.12045)
- [τ²-bench](https://arxiv.org/abs/2506.07982)
- [RealUserSim](https://arxiv.org/abs/2605.20204)（2026 preprint）
- [OSWorld](https://arxiv.org/abs/2404.07972)
- [AppWorld](https://arxiv.org/abs/2407.18901)
- [OpenAI: Why We No Longer Evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- [Anthropic: Eval Awareness in BrowseComp](https://www.anthropic.com/engineering/eval-awareness-browsecomp)
- [METR: Recent Reward Hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
