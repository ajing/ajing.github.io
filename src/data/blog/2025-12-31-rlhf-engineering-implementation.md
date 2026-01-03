---
author: Jing Lu
pubDatetime: 2025-12-31T00:00:00Z
modDatetime: 2025-12-31T00:00:00Z
title: "RLHF from an Engineering Perspective: PPO, GRPO, DPO, and Tool-Use Implementation"
featured: true
draft: false
tags:
  - RLHF
  - PPO
  - DPO
  - GRPO
  - ML
  - Engineering
description: "A practical engineering guide to RLHF implementation—understanding logprobs, KL computation, PPO, GRPO, DPO, and tool-use from the code perspective."
---

> **Goal:** When you pick up an RLHF/reasoning/tool-calling training codebase, you should be able to quickly answer:
> - How is KL computed? Does it go in the reward or the loss?
> - How are per-token / per-seq losses aggregated? What's the GRPO/Dr.GRPO debate about?
> - What is the PPO value head fitting? Why the `0.5` in `0.5*(x-y)^2`?
> - Why is DPO "not a policy gradient estimator" but still trained via gradient descent?
> - Where does tool-use trajectory data come from?

---

## The Minimal Training Skeleton

Almost all post-training ultimately reduces to:

```python
# per_token_loss: (B, L) or (B*G, L)
# completion_mask: (B, L) or (B*G, L)   # 1 for response tokens, 0 for prompt/pad

loss = (per_token_loss * completion_mask).sum(...) / normalizer
loss.backward()
opt.step()
```

All confusion about "normalization strategy," "length bias," and "Dr.GRPO fixed normalizer" boils down to:

**What is `normalizer`? How is `mask` defined? What shape is `per_token_loss`?**

---

## logprob / logits / NLL: The Alignment Problem

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

**Pitfall:** Your mask's L is the completion length, but T-1 includes the prompt. You need `completion_mask` aligned to the `(B, T-1)` axis.

---

## KL: Three Different Things in Code

### 1. On-Sample Trajectory KL (Most Common)

```python
kl_token = token_logprobs - ref_token_logprobs      # (B, L)
```

### 2. Full Distribution KL (Expensive)

```python
kl_full = (probs * (logprobs - ref_logprobs)).sum(-1)   # (B, L)
```

### 3. approx_kl for Logging

```python
approx_kl = 0.5 * ((new_logp - old_logp) ** 2).mean()
```

---

## PPO: Policy Loss + Value Loss

### Why `0.5*(v_pred-target)^2`?

The `0.5` is purely engineering convention: it makes the gradient become `(v_pred-target)` instead of `2*(v_pred-target)`.

### PPO Policy Loss: ratio + clip + min

```python
ratio = torch.exp(new_logp - old_logp)             # (B, L)
pg1 = -advantages * ratio
pg2 = -advantages * torch.clamp(ratio, 1-eps, 1+eps)
pg_loss_tok = torch.max(pg1, pg2)
policy_loss = masked_mean(pg_loss_tok, completion_mask)
```

**Key intuition:** min/clip ensures that when updates are too large, the objective flattens and no longer encourages moving in the same direction.

---

## GRPO: Group Comparison to Bypass Value Function

Two key differences from PPO:

1. Advantage is **bandit-style** (one scalar per completion) then broadcast to all tokens
2. KL penalty is often **added directly to per-token loss**

### Group-Level Advantage Normalization

```python
mu = rewards.view(B, G).mean(1)
std = rewards.view(B, G).std(1)
adv = (rewards - mu.repeat_interleave(G)) / (std.repeat_interleave(G) + 1e-4)
```

**Problem with "only one success in group":** Extreme reward distribution → unstable `std` statistics.

---

## DPO: Offline Contrastive Loss

```python
pi_lr  = policy_chosen_logps - policy_rejected_logps
ref_lr = ref_chosen_logps    - ref_rejected_logps
logits = pi_lr - ref_lr
loss = -F.logsigmoid(beta * logits).mean()
```

**β controls:** "preference-driving force vs staying-close-to-reference regularization."

---

## Tool Use: Data to Objective

### Where Does Data Come From?

1. **Toolformer-style self-label:** Model proposes "tool should be called here"
2. **ToolBench-style generation:** Large-scale trajectory generation
3. **High-quality human trajectories:** For critical domains

### Common Engineering Pitfalls

- Does tool output enter context? Does it pollute training signal?
- Tool call token masking: Which tokens are you supervising?
- How to score failed trajectories: Sparse reward → credit assignment difficulty

---

## Algorithm Comparison

| Aspect | PPO | GRPO | DPO |
|--------|-----|------|-----|
| **Input** | Prompt + completions + reward | Prompt + G completions + reward | Preference pairs |
| **Advantage** | Per-token (GAE or MC) | Per-completion, broadcast | Implicit |
| **KL Handling** | Fold into reward or loss | Usually add to loss | Implicit |
| **Value Head** | Yes | No | No |
| **Key Hyperparams** | `eps`, `vf_coef`, `gamma` | `G`, `beta`, normalization | `beta` |

---

## Debug Checklist

### Monitor These 6 Scalars

- `mean reward` (per prompt, per batch)
- `std reward` (especially GRPO within-group std)
- `approx_kl` (is it continuously rising?)
- `clip_frac` (PPO clip trigger rate)
- `entropy` (is it collapsing?)
- `grad_norm` (is it exploding?)

### Common Symptoms → Causes

| Symptom | Likely Cause |
|---------|--------------|
| Reward rises but usability drops | Over-optimization / judge bias |
| KL spikes | Learning rate too high, β too small |
| GRPO unstable | Extreme within-group std, G too small |
| Tool-use hallucination | Tool result not in context but model fabricates |

---

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
2. Shao, Z., et al. "DeepSeekMath." arXiv:2402.03300 (2024)
3. Rafailov, R., et al. "Direct Preference Optimization." NeurIPS 2023
4. Schick, T., et al. "Toolformer." arXiv:2302.04761 (2023)

