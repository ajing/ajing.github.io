---
author: Jing Lu
pubDatetime: 2025-12-31T00:00:00Z
title: "Post-Training Is Not 'One Algorithm': Objective Functions and Implementation Essentials for PPO / DPO / GRPO"
featured: true
draft: false
tags:
  - RLHF
  - PPO
  - DPO
  - GRPO
  - ML
description: "Reading notes from Nathan Lambert's RLHF book—understanding post-training as an engineering pipeline: data → reward proxy → optimization → evaluation → guardrails."
---

> Reading notes from Nathan Lambert's "Reinforcement Learning from Human Feedback (RLHF)." The book helped me build a clearer mental model of post-training—not as a single algorithm, but as an engineering pipeline: data → reward proxy → optimization → evaluation → guardrails.
>
> **Notation**: prompt is $x$, completion is $y$, policy model is $\pi_\theta(y\mid x)$, reference model is $\pi_{\text{ref}}$, reward/scoring function is $r(x,y)$.

---

## 0) A "Production-Ready" Pipeline Diagram

```
SFT -> Preference data (pairs / rankings) -> RM/Judge/Verifier -> Optimize (PPO/GRPO/DPO/...) -> Eval & Iterate
                       |                                               |
                       +-------------------- guardrails (KL/NLL/len/...) +
```

**Key insight**: What you actually want is **usable behavior in production**; but what you can optimize is often just some **proxy objective**, so the system naturally evolves toward "needing guardrails."

---

## 1) Canonical RLHF: Reward - KL (What the Objective Function Looks Like)

The book consistently uses this form of RLHF: maximize expected reward over the data distribution while using KL to pull the policy back toward the reference model (style/distribution constraint).

$$
\max_{\pi} \ \mathbb{E}_{x\sim D}\mathbb{E}_{y\sim \pi(\cdot|x)}[r(x,y)] - \beta D_{\mathrm{KL}}(\pi(\cdot|x)\Vert \pi_{\mathrm{ref}}(\cdot|x))
$$

**Intuition**: $\beta$ is the knob for "how far you dare to stray from SFT"; it's not just a mathematical term—it's more like the master switch for "capability vs style/stability" in your product.

---

## 2) PPO: The Core Isn't "Being Stronger"—It's That Clipping Makes Updates More Stable

PPO's key is **importance ratio + clipping**: for the same batch of data sampled by the old policy $\pi_{\theta_\text{old}}$, use the ratio to correct when updating the new policy, and use clip to limit single-update magnitude, preventing training collapse when the ratio deviates too far from 1.

**Classic PPO clipped surrogate:**

$$
J(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]
$$

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}
$$

In language models, the common implementation is **per-token** (easier to compute with logprobs).

### Minimal Working PPO Pseudocode (LM Version)

```python
# given prompts x
y = sample(pi_old, x)                         # rollout / completion
r = reward_model_or_verifier(x, y)            # scalar per sequence (or per step)

logp_new = logprob(pi_theta, x, y)            # [B, L]
logp_old = logprob(pi_old,   x, y)            # [B, L]
ratio = exp(logp_new - logp_old)              # [B, L]

A = compute_advantage(r, baseline=V(x,y) or batch_norm)  # token- or seq-level
pg = -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))

kl = mean(logp_new - logprob(pi_ref, x, y))   # MC estimate of reverse KL
loss = pg + beta*kl + vf_coef*value_loss + other_regularizers
loss.backward(); opt.step()
```

> **The easiest pitfalls in engineering aren't the formulas**—they're: **how advantage is defined (token vs sequence), whether KL goes in reward or loss, how many gradient steps per batch of data**. These significantly change "what training produces."

---

## 3) KL Regularization: The "Numerical Preference" of Reverse KL and Implementation

The book treats KL control as one of the core guardrails of post-training, explaining the common **reverse KL to reference**: when the new policy assigns high probability in regions where the reference model has low probability, it gets heavily penalized (numerically more "conservative").

Implementation often uses sampling to approximate the expectation:

$$
D_{\mathrm{KL}}(P\Vert Q)=\mathbb{E}_{x\sim P}[\log P(x)-\log Q(x)]
$$

### Forward KL vs Reverse KL Intuition

| | Forward KL | Reverse KL |
|---|---|---|
| **Penalizes** | $P_\theta$ assigning low prob where $P_{ref}$ is high | $P_\theta$ assigning high prob where $P_{ref}$ is low |
| **Behavioral tendency** | Mode-covering (tries to cover all modes) | Mode-seeking (tends to converge to single mode) |
| **Practical effect** | More "exploratory," may produce strange outputs | More "conservative," tends toward safe answers |

RLHF commonly uses **Reverse KL** because we want to avoid the model "making things up" (assigning high probability where reference thinks it's impossible).

---

## 4) DPO: Folding "RLHF + KL" into an Offline Contrastive Loss

DPO's position in the book is clear: it **doesn't do online rollouts, doesn't separately train an RM**, directly doing gradient descent on preference pairs.

**DPO (Bradley–Terry form) core loss:**

$$
L_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}})=
-\mathbb{E}_{(x,y_c,y_r)\sim D}
\left[
\log \sigma\left(
\beta \log\frac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)}
-
\beta \log\frac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)}
\right)\right]
$$

### Minimal Working DPO Pseudocode

```python
# batch of (x, y_chosen, y_rejected)
lc = sum_token_logp(pi_theta, x, y_chosen)
lr = sum_token_logp(pi_theta, x, y_rejected)
lcref = sum_token_logp(pi_ref, x, y_chosen)
lrref = sum_token_logp(pi_ref, x, y_rejected)

delta = beta * ((lc - lcref) - (lr - lrref))
loss = -log(sigmoid(delta)).mean()
loss.backward(); opt.step()
```

> The book especially emphasizes: DPO appears to be "directly training policy," but essentially it's still learning reward structure (hence the **"Your LM is secretly a reward model"** statement).

---

## 5) GRPO: Like PPO, But Using Group Comparison to Bypass Value Function

GRPO (used in DeepSeekMath) can be viewed as a PPO-style surrogate loss, but it **avoids training a value function**: by sampling multiple completions per prompt and doing within-group normalization to estimate advantage.

**GRPO objective:**

$$
J(\theta)=\frac{1}{G}\sum_{i=1}^{G}
\Big[\min(\rho_i A_i,\ \text{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i)
-\beta D_{\mathrm{KL}}(\pi_\theta\Vert \pi_{\text{ref}})\Big]
$$

$$
A_i=\frac{r_i-\text{mean}(r_{1:G})}{\text{std}(r_{1:G})}
$$

### GRPO vs PPO: Why Use Group Comparison?

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Baseline** | Value function $V(s)$ | Within-group mean $\bar{r}$ |
| **Extra model** | Needs Critic training | Not needed |
| **Memory overhead** | High (storing value head) | Low |
| **Use case** | Complex multi-step decisions | Bandit-style (single generation) |

---

## 6) Algorithm Selection Guide

```
                    Have online environment?
                              |
                   +----------+----------+
                   |                     |
                  Yes                    No
                   |                     |
              Have value head           DPO
              training budget?     (offline preference pairs)
                   |
           +-------+-------+
           |               |
          Yes              No
           |               |
          PPO            GRPO
       (full RL)    (group comparison)
```

| Scenario | Recommended Algorithm | Reasoning |
|----------|----------------------|-----------|
| Lots of preference pairs, want fast iteration | DPO | Simple implementation, no rollout needed |
| Have verifier/env feedback, sufficient resources | PPO | Most flexible, can do multi-step optimization |
| Have verifier, but memory-constrained | GRPO | No value head needed |
| Math/code with correct answers | GRPO | Verifier easy to define |

---

## Final 3 Takeaways

1. **PPO's value is stable updates**: ratio + clipping prevents "aggressive optimizer" from blowing up the model.

2. **DPO folds RLHF into offline contrastive learning**: Simple implementation, fast iteration, but learning space is bounded by offline data.

3. **GRPO replaces value function with group comparison**: Lower memory / fewer components, but you need to take reward design and group sampling strategy more seriously.

---

## References

1. Lambert, N. "Reinforcement Learning from Human Feedback." (2024)
2. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
3. Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023
4. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300 (2024)

