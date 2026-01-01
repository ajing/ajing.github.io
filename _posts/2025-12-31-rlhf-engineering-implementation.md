---
layout: post
title: "RLHF from an Engineering Perspective: PPO, GRPO, DPO, and Tool-Use Implementation"
---

# RLHF from an Engineering Perspective: PPO, GRPO, DPO, and Tool-Use Implementation

> **Goal:** When you pick up an RLHF/reasoning/tool-calling training codebase, you should be able to quickly answer:
> - How is KL computed? Does it go in the reward or the loss?
> - How are per-token / per-seq losses aggregated? What's the GRPO/Dr.GRPO debate about?
> - What is the PPO value head fitting? Why the `0.5` in `0.5*(x-y)^2`?
> - Why is DPO "not a policy gradient estimator" but still trained via gradient descent? What does β control?
> - Where does tool-use trajectory data come from? How do SFT / DPO / RL approaches differ?

---

## 0) The Minimal Training Skeleton: Everything Is a "Masked Loss"

Almost all post-training ultimately reduces to:

```python
# per_token_loss: (B, L) or (B*G, L)
# completion_mask: (B, L) or (B*G, L)   # 1 for response tokens, 0 for prompt/pad

loss = (per_token_loss * completion_mask).sum(...) / normalizer
loss.backward()
opt.step()
```

All your confusion about "normalization strategy," "length bias," "group-level standardization," and "Dr.GRPO fixed normalizer" boils down to:

**What is `normalizer`? How is `mask` defined? What shape is `per_token_loss`? Is advantage per-token or broadcast?**

---

## 1) logprob / logits / NLL: The Alignment Problem That's Easiest to Get Wrong

### 1.1 logits → logprobs → token_logprobs (gather)

Typical forward output:

- `logits`: `(B, T, V)` where T = prompt + completion total length
- `tokens`: `(B, T)` same-length token ids

**Next-token alignment (critical):**

```python
logits = model(tokens[:, :-1]).logits          # (B, T-1, V)
logprobs = logits.log_softmax(dim=-1)          # (B, T-1, V)

next_tokens = tokens[:, 1:]                    # (B, T-1)
token_logprobs = logprobs.gather(
    dim=-1,
    index=next_tokens.unsqueeze(-1)            # (B, T-1, 1)
).squeeze(-1)                                   # (B, T-1)
```

**Pitfall:** Your mask's L is the completion length, but T-1 here includes the prompt. So you need `completion_mask` aligned to the `(B, T-1)` axis (usually all zeros in the prompt region).

### 1.2 NLL / Cross-Entropy vs "log-prob"

- `NLLLoss`/`CrossEntropyLoss`: Supervised learning perspective (teacher forcing)
- `logprob`: The core quantity for building ratio / policy gradient in RL

In practice, you'll use both:
- SFT phase: CE/NLL
- RL phase: logprob for PPO/GRPO per-token loss, plus **SFT-mix** to prevent degradation (see steering losses below)

---

## 2) KL: Three Different Things in Code (Must Distinguish)

### 2.1 "On-Sample Trajectory KL" (Most Common, Cheapest)

Compare new/old logprobs only on sampled tokens:

```python
# token_logprobs: (B, L) for sampled completion tokens
# ref_token_logprobs: (B, L)
kl_token = token_logprobs - ref_token_logprobs      # (B, L)
```

This is often used as a **per-token approximation of KL penalty**. Advantages: cheap, directly usable in loss.

> **Q: "Will positive and negative cancel to 0?"**
> Not guaranteed. You're comparing on the sampling path, and the expectation measure doesn't have a "symmetric cancellation" structure.

### 2.2 "Full Distribution KL" (Expensive, Closer to True KL)

```python
# logprobs/ref_logprobs: (B, L, V)
kl_full = (probs * (logprobs - ref_logprobs)).sum(-1)   # (B, L)
```

This requires expectation over vocabulary dimension V—expensive, not always used in RLHF.

### 2.3 "approx_kl for Logging" (Common in PPO)

```python
approx_kl = 0.5 * ((new_logp - old_logp) ** 2).mean()
```

This is a second-order approximation from certain derivations, commonly used for monitoring—not necessarily equivalent to the KL in your loss.

---

## 3) PPO: Policy Loss + Value Loss (Including the 0.5 and G Questions)

### 3.1 What Is the Value Head Fitting?

In PPO, the critic predicts `V(s_t)`: the expected future return from the current token-prefix state.

- **What is G in `A_t = G_t - V(s_t)`?**
  G_t is the **return** (cumulative reward from t onwards)—can be Monte Carlo return or GAE target.

- **Is G the same as Q?**
  More precisely: **G_t is a Monte Carlo sample estimate of Q(s_t, a_t)** (with action fixed to the sampled token).

### 3.2 Why `0.5*(v_pred-target)^2`?

The square is MSE; `0.5` is purely engineering convention: it makes the gradient become `(v_pred-target)` instead of `2*(v_pred-target)`.

### 3.3 Simplest PPO Critic (No GAE)

```python
# rewards: (B, L)  post-KL per-token reward (or raw reward then subtract KL)
# done_mask: (B, L) 1 at terminal else 0
returns = torch.zeros_like(rewards)
running = torch.zeros(B, device=rewards.device)

for t in reversed(range(L)):
    running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
    returns[:, t] = running

targets = returns                                 # (B, L)
vf_loss_tok = 0.5 * (values - targets)**2         # (B, L)
value_loss = masked_mean(vf_loss_tok, completion_mask)

advantages = (targets - values).detach()           # (B, L)
```

### 3.4 PPO Policy Loss: ratio + clip + min

**Why can't you just use `clipped * A`?**

```python
ratio = torch.exp(new_logp - old_logp)             # (B, L)
pg1 = -advantages * ratio
pg2 = -advantages * torch.clamp(ratio, 1-eps, 1+eps)
pg_loss_tok = torch.max(pg1, pg2)                  # Note: max (equivalent to min surrogate)
policy_loss = masked_mean(pg_loss_tok, completion_mask)
```

**Key intuition:** min/clip isn't "arbitrary truncation"—it ensures that **when updates are too large, the objective flattens and no longer encourages moving in the same direction** (both A>0 and A<0 cases need handling, so you can't just write one clipped version).

---

## 4) GRPO / Dr.GRPO: Core Differences Are in Advantage Shape and Normalization

The two most important points in GRPO:

1. Advantage is **bandit-style** (one scalar per completion) then broadcast to all tokens
2. KL penalty is often **added directly to per-token loss**, rather than folded into reward (both approaches exist)

### 4.1 What Are the Initial Dimensions of Rewards?

If:
- `B`: number of prompts
- `G`: number of completions sampled per prompt
- `L`: completion length (padded to uniform)

Common storage formats:
- `rewards`: `(B*G,)` one final reward per completion (e.g., answer correct/wrong)
- Or `rewards_token`: `(B*G, L)` per-token reward (common when treating KL as per-token reward)

This line:

```python
mean_grouped_rewards = rewards.view(-1, G).mean(dim=1)
```

indicates `rewards` is `(B*G,)` (or at least reshapeable to `(B, G)`).

### 4.2 Group-Level Advantage Normalization: Why Is "Only One Success in the Group" Problematic?

Common GRPO normalization:

```python
mu = rewards.view(B, G).mean(1)          # (B,)
std = rewards.view(B, G).std(1)          # (B,)
adv = (rewards - mu.repeat_interleave(G)) / (std.repeat_interleave(G) + 1e-4)  # (B*G,)
```

When only one of G samples per prompt succeeds:
- Reward distribution is extreme → `std` statistics are unstable
- Normalization compresses that successful sample's "absolute value" into a "relative z-score," potentially causing **learning signal dilution or directional jitter** on some prompts

This is what "normalize losses or advantages at batch level instead of group level" means: replace `mu/std` from "per-prompt group statistics" with "entire batch statistics" to reduce instability from extreme group distributions (but sacrificing some local structure of within-prompt relative ranking).

### 4.3 Three Loss Aggregation Strategies (Strategy 2 vs 3: Are They the Same?)

Most engineering-focused comparison:

- **Strategy 1 (per-seq):** Each completion has equal weight

```python
loss = ((tok_loss * mask).sum(1) / mask.sum(1)).mean()
```

- **Strategy 2 (per-token):** Each token has equal weight (longer samples have more influence)

```python
loss = (tok_loss * mask).sum() / mask.sum()
```

- **Strategy 3 (fixed-length, Dr.GRPO):** Denominator is constant `max_gen_len` (unified scale)

```python
loss = (tok_loss * mask).sum(1) / max_gen_len
loss = loss.mean()
```

**Strategy 2 vs 3 are not equivalent:** When length distribution varies greatly within a batch, gradient scale and stability differ; Strategy 3 is more like "fixing per-sample per-token scale" then letting longer samples naturally contribute more due to more tokens.

### 4.4 KL in GRPO: Added to Loss vs Folded into Reward

This approach:

```python
per_token_loss = pg_loss_max + beta * per_token_kl
```

is "added to loss."

PPO/RLOO commonly fold `-beta*kl` into reward, then compute return/advantage. Both work, but must be consistent—otherwise you'll contradict yourself on "what advantage means."

---

## 5) DPO: Simplest Engineering Implementation, But Two Things Must Be Clear

Two key conclusions stated as **bold sentences**:

> **DPO is not a policy-gradient estimator (doesn't rely on rollouts + advantage estimation), but it's still trained via gradient descent on parameters.**
>
> **More importantly: DPO learns an "implicit reward model" (log-ratio form), not "completely bypassing reward."**

### 5.1 Most Common DPO Implementation (Pairwise)

```python
# policy_logps: sum of token logprob for chosen/rejected completion
pi_lr  = policy_chosen_logps - policy_rejected_logps
ref_lr = ref_chosen_logps    - ref_rejected_logps
logits = pi_lr - ref_lr
loss = -F.logsigmoid(beta * logits).mean()
```

### 5.2 What Is β (The "Fitting vs KL Balance")

At the implementation level:
- Larger β: More strongly pushes apart chosen vs rejected logprob margin (more aggressive preference fitting)
- Smaller β: More conservative updates, staying closer to reference (equivalent to stronger KL constraint effect)

**β fundamentally controls "preference-driving force vs staying-close-to-reference regularization."**

---

## 6) Tool Use & Function Calling: "Most Practical Three-Stage" From Data to Objective

### 6.1 Where Does Data Come From (Cheapest First: Synthetic/Bootstrap)

- **Toolformer-style self-label:** Model proposes "tool should be called here" and generates trace
- **ToolBench-style generation:** Large-scale generation of "tool call trajectories + task descriptions + execution results"
- **Small amounts of high-quality human trajectories:** For critical domains (safety/privacy/core product features)

### 6.2 How to Set Objectives (Easy to Hard)

1. **SFT on trajectories:** Learn JSON schema, call positions, parameter formats

2. **Preference optimization (DPO/RLAIF):** Rank on "should call or not, call order, whether to hallucinate tool output," etc.

3. **RL with env feedback:** Most natural for multi-step agents
   - Reward directly from: task success / constraint satisfaction / tool execution pass/fail

### 6.3 Most Common Engineering Pitfalls

- **Does tool output enter context?** If so, does it pollute training signal?
- **Tool call token masking:** Which tokens are you actually supervising (planning text? call JSON? answer?)
- **How to score failed trajectories:** Sparse reward causes credit assignment difficulty → often needs shaped reward or rubric judge

---

## 7) Evaluation: "5 Variables You Must Control" From an Engineering Perspective

An actionable checklist (root causes of unreliable external comparisons are almost always here):

1. **Prompt template** (especially multiple-choice/format checking)
2. **Sampling strategy** (temperature, top-p, n samples, majority vote/pass@k)
3. **Token budget / inference-time scaling** (CoT length, tool call count)
4. **Answer extractor** (regex/boxed/end-of-response phrase)
5. **Contamination risk** (does training data include benchmark/approximate rewrites)

> **Reality:** Same model with slightly different prompt or extraction can have dramatically different scores; "cross-comparison" without controlled variables is extremely noisy.

---

## 8) Over-Optimization & Steering Losses: What You'll Actually Add to Total Loss

The "most common recipe":

```python
total_loss = (
    rl_loss
  + vf_coef * value_loss                 # PPO only
  + kl_coef * kl_penalty                 # Added to loss or folded into reward
  + sft_coef * sft_nll_loss              # Prevent language degradation/mode collapse
  + entropy_coef * entropy_bonus         # Prevent collapse (optional)
  + length_coef * length_penalty         # Suppress verbose gaming (optional)
)
```

These terms exist not as "black magic" but to combat the same problem:

**Reward hacking / over-optimization causing generalization degradation, style drift, factuality decline, verbosity/sycophancy.**

---

## 9) Debug Checklist: What to Check First When Running PPO/GRPO/DPO/Tool-Use

### 9.1 Check Shape and Mask First

- Are `token_logprobs`, `completion_mask`, `values` strictly the same shape?
- Is prompt region all zeros? Is pad region all zeros?
- Can normalizer denominator be zero? (`clamp_min(1)`)

### 9.2 Monitor 6 Scalars (More Useful Than Watching Loss)

- `mean reward` (per prompt, per batch)
- `std reward` (especially GRPO within-group std)
- `approx_kl` (is it continuously rising?)
- `clip_frac` (PPO clip trigger rate)
- `entropy` (is it collapsing?)
- `grad_norm` (is it exploding?)

### 9.3 Common Symptoms → Common Causes

| Symptom | Likely Cause |
|---------|--------------|
| Reward rises but usability drops | Over-optimization / judge bias / length gaming |
| KL spikes | Learning rate too high, β/kl_coef too small, inconsistent reference update strategy |
| GRPO unstable | Extreme within-group std, G too small, inappropriate normalization |
| Tool-use hallucination | Trained on tool output as ground truth, tool result not in context but model learns to fabricate |

---

## 10) Algorithm Comparison Table

| Aspect | PPO | GRPO | DPO |
|--------|-----|------|-----|
| **Input** | Prompt + sampled completions + reward | Prompt + G completions/prompt + reward | Prompt + chosen/rejected pairs |
| **Key Shape** | `(B, L)` per-token | `(B*G, L)` or `(B, G, L)` | `(B,)` pairwise |
| **Advantage** | Per-token (GAE or MC) | Per-completion (bandit), broadcast | Implicit in log-ratio |
| **KL Handling** | Fold into reward or add to loss | Usually add to loss | Implicit via reference |
| **Value Head** | Yes (critic) | No | No |
| **Key Hyperparams** | `eps`, `vf_coef`, `gamma`, `gae_lambda` | `G`, `beta`, normalization strategy | `beta` |
| **Common Pitfalls** | Value head fitting, GAE computation | Group normalization instability | β tuning, reference drift |

---

## Conclusion: The Unified Engineering Perspective

You can compress the "engineering core" of this entire topic into one sentence:

> **Post-training isn't about choosing algorithm names—it's about choosing: reward/advantage definition, KL implementation and placement, and the statistical properties of loss aggregation and normalization.**

When debugging or implementing any RLHF variant, always trace back to these three questions:
1. What is my advantage/reward signal, and at what granularity?
2. How and where is KL computed and applied?
3. What are my masking, aggregation, and normalization choices doing to gradient statistics?

Answer these, and you'll understand why your training is (or isn't) working.

---

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347 (2017).
2. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv preprint arXiv:2402.03300 (2024).
3. Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.
4. Schick, T., et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." arXiv preprint arXiv:2302.04761 (2023).
5. Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." arXiv preprint arXiv:2307.16789 (2023).

