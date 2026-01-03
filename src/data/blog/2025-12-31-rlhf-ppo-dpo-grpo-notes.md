---
author: Jing Lu
pubDatetime: 2025-12-31T00:00:00Z
title: "Post-Training Is Not 'One Algorithm': Objective Functions and Implementation Essentials for PPO / DPO / GRPO"
featured: true
draft: false
tags:
  - AI
  - RLHF
  - ML Engineering
  - LLM
description: "Reading notes on RLHF covering PPO, DPO, and GRPO—understanding post-training as an engineering pipeline rather than a single algorithm."
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

This also explains why many systems can compute KL by just calculating logprobs twice (no explicit summation needed).

### Forward KL vs Reverse KL Intuition

| | Forward KL $D_{KL}(P_{ref} \| P_\theta)$ | Reverse KL $D_{KL}(P_\theta \| P_{ref})$ |
|---|---|---|
| **Penalizes** | $P_\theta$ assigning low prob where $P_{ref}$ is high | $P_\theta$ assigning high prob where $P_{ref}$ is low |
| **Behavioral tendency** | Mode-covering (tries to cover all modes) | Mode-seeking (tends to converge to single mode) |
| **Practical effect** | More "exploratory," may produce strange outputs | More "conservative," tends toward safe answers |

RLHF commonly uses **Reverse KL** because we want to avoid the model "making things up" (assigning high probability where reference thinks it's impossible).

---

## 4) DPO: Folding "RLHF + KL" into an Offline Contrastive Loss (Engineering-Friendly)

DPO's position in the book is clear: it **doesn't do online rollouts, doesn't separately train an RM**, directly doing gradient descent on preference pairs; meanwhile it corresponds to a closed-form optimal solution of the "KL-constrained RLHF objective" (given data and $\beta$).

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

It can also be interpreted as learning an "implicit reward" (log-ratio structure).

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

> The book especially emphasizes a common misconception: DPO appears to be "directly training policy," but essentially it's still learning reward structure (hence the **"Your LM is secretly a reward model"** statement).

### DPO's Implicit Reward Interpretation

From DPO's derivation, we can extract the implicit reward:

$$
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

where $Z(x)$ is the partition function. This means:
- A DPO-trained model **is itself a reward model**
- You can use a trained DPO model to score new completions

---

## 5) GRPO: Like PPO, But Using Group Comparison to Bypass Value Function

GRPO (used in DeepSeekMath and other work) can be viewed as a PPO-style surrogate loss, but it **avoids training a value function**: by sampling multiple completions per prompt and doing within-group normalization to estimate advantage.

**GRPO objective (group-aggregated):**

$$
J(\theta)=\frac{1}{G}\sum_{i=1}^{G}
\Big[\min(\rho_i A_i,\ \text{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i)
-\beta D_{\mathrm{KL}}(\pi_\theta\Vert \pi_{\text{ref}})\Big]
$$

$$
A_i=\frac{r_i-\text{mean}(r_{1:G})}{\text{std}(r_{1:G})}
$$

### Minimal Working GRPO Pseudocode

```python
# For each prompt x, sample G completions
y_group = [sample(pi_old, x) for _ in range(G)]    # G completions per prompt
r_group = [reward(x, y) for y in y_group]          # G rewards

# Group-level advantage normalization
mu = mean(r_group)
std = std(r_group) + 1e-8
advantages = [(r - mu) / std for r in r_group]     # z-score within group

# PPO-style loss for each completion
for y, A in zip(y_group, advantages):
    logp_new = logprob(pi_theta, x, y)
    logp_old = logprob(pi_old, x, y)
    ratio = exp(logp_new - logp_old)
    
    pg1 = -A * ratio
    pg2 = -A * clip(ratio, 1-eps, 1+eps)
    pg_loss = max(pg1, pg2)                        # element-wise max
    
    kl = logp_new - logprob(pi_ref, x, y)
    loss = pg_loss + beta * kl
```

The book also mentions implementation details: GRPO commonly adds KL **directly to the loss** (rather than modifying reward first), which differs from traditional PPO.

### GRPO vs PPO: Why Use Group Comparison?

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Baseline** | Value function $V(s)$ | Within-group mean $\bar{r}$ |
| **Extra model** | Needs Critic training | Not needed |
| **Memory overhead** | High (storing value head) | Low |
| **Variance** | GAE can control | Depends on group size G |
| **Use case** | Complex multi-step decisions | Bandit-style (single generation) |

---

## 6) Preference Data: The Most Powerful Fuel, Also the Most Hidden Bias Amplifier

The book dedicates a section to "bias in data collection," calling out **prefix bias, sycophancy, verbosity, formatting**, etc.—these often aren't written in labeling guidelines but get learned very firmly by models.

### Common Data Bias Types

| Bias Type | Manifestation | Consequence |
|-----------|---------------|-------------|
| **Length bias** | Longer answers more likely chosen as preferred | Model becomes verbose, information density drops |
| **Format bias** | Markdown/lists more likely to win | Over-formatting, bullet points even for simple questions |
| **Sycophancy** | Agreeing with user more likely chosen | Model becomes "pleasing," afraid to correct errors |
| **Position bias** | First/last option more likely chosen | Evaluation results unstable |
| **Verbosity ≠ Quality** | Detailed ≠ correct | Reward hacking |

**My engineering conclusion:**
For preference pair data, the difference often isn't in quantity but in whether these biases are systematically addressed (e.g., UI display, labeling workflow, length control, penalties for "flattery/fluff").

> **Real case**: In an internal judge, discovered "longer, more template-like answers win more easily," causing DPO-trained model output information density to drop; had to add length-control / information density constraints to recover.

---

## 7) Evaluation: Why "All Scores Are Rising" But You Still Don't Dare Ship

The book's attitude toward evaluation is realistic: evaluation evolves with training objectives, and **prompt/format** can take the same model's performance from "okay" to "near zero" (extremely sensitive).

### The Triple Dilemma of Evaluation

1. **Prompt sensitivity**: Same model, different prompt template, scores can differ 20%+
2. **Metric gaming**: Optimizing benchmark scores ≠ real capability improvement
3. **Distribution shift**: Training distribution vs evaluation distribution vs real user distribution—all three inconsistent

### Internal vs External Evaluation

| | Internal Evaluation | External Evaluation |
|---|---|---|
| **Purpose** | Hillclimbing, guide iteration | Comparison, release decisions |
| **Characteristics** | Controllable variables, reproducible | Opaque configuration, high error |
| **Risk** | Overfitting internal benchmark | Not reproducible, high noise |

### LLM-as-a-Judge Engineering Tips

Although commonly used (including for generating preference data), note:

```python
# Common tricks to reduce variance
judge_config = {
    "temperature": 0,           # Deterministic output
    "max_tokens": 1,            # Only want score, not explanation
    "logprobs": True,           # Use logprob rather than argmax
}

# Position debiasing
score_AB = judge(response_A, response_B)
score_BA = judge(response_B, response_A)
final_score = (score_AB - score_BA) / 2  # Cancel position bias
```

---

## 8) Over-Optimization: Not an Occasional Bug, But a Default Risk

The book gives a definition I really like:

> **When you optimize hard on a proxy as if it were the target, the "true objective" first improves then degrades** (classic Goodhart's Law).

### Typical Over-Optimization Symptoms

| Symptom | Cause | Mitigation |
|---------|-------|------------|
| **Fixed phrases** | Certain phrases overvalued by RM | Diversity regularization, entropy bonus |
| **Repetition/Hedging** | Safe outputs score high | Penalize repeated n-grams |
| **Sycophancy** | Agreeing with user scores high | Dedicated sycophancy detector |
| **Excessive refusal** | Refusing is safer than being wrong | Balance helpfulness vs harmlessness |
| **Length gaming** | Long answers score high | Length penalty term |

### Guardrails Aren't Decoration—They're Survival Necessities

```python
total_loss = (
    rl_loss                              # Main objective
    + beta * kl_penalty                  # Don't deviate too far from reference
    + sft_coef * sft_loss                # Maintain language capability
    + length_coef * length_penalty       # Control length
    + entropy_coef * entropy_bonus       # Maintain diversity
    + format_coef * format_penalty       # Format constraints
)
```

---

## 9) One-Page Engineering Checklist: What to Monitor When Running PPO/DPO/GRPO

### Training Side (Observable)

| Metric | Focus | Alert Threshold |
|--------|-------|-----------------|
| `mean_reward` | Is it continuously rising | Sudden drop or saturation |
| `reward_std` | Is distribution healthy | Too small (collapse) or too large (unstable) |
| `kl_divergence` | Deviation from reference | > 10-15 usually problematic |
| `clip_fraction` | PPO clip trigger rate | > 30% may mean learning rate too high |
| `entropy` | Output diversity | Continuous decline = collapse |
| `grad_norm` | Gradient health | Sudden spike = instability |

### Evaluation Side (Reproducible)

- [ ] Fixed prompting template & sampling parameters (temperature, top-p, token budget)
- [ ] Private "regression set" (subset of real product traffic)
- [ ] Human eval / A-B test (don't trust a single score)
- [ ] Variance estimation across multiple seeds

### Data Side (Bias Governance)

- [ ] Length/format bias countermeasures (e.g., length-controlled evaluation, format perturbation robustness)
- [ ] Dedicated sycophancy data/rules/discriminator
- [ ] Does labeling UI introduce position bias
- [ ] Regular audit of labeling quality

---

## 10) Algorithm Selection Guide

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

### When to Choose Which?

| Scenario | Recommended Algorithm | Reasoning |
|----------|----------------------|-----------|
| Lots of preference pairs, want fast iteration | DPO | Simple implementation, no rollout needed |
| Have verifier/env feedback, sufficient resources | PPO | Most flexible, can do multi-step optimization |
| Have verifier, but memory-constrained | GRPO | No value head needed |
| Math/code with correct answers | GRPO | Verifier easy to define |
| Open-ended generation, subjective preferences | DPO | Leverages human preference data |

---

## Final 3 Takeaways (Easy to Remember)

1. **PPO's value is stable updates**: ratio + clipping prevents "aggressive optimizer" from blowing up the model.

2. **DPO folds RLHF into offline contrastive learning**: Simple implementation, fast iteration, but learning space is bounded by offline data.

3. **GRPO replaces value function with group comparison**: Lower memory / fewer components, but you need to take reward design and group sampling strategy more seriously.

---

## Appendix: Objective Function to Implementation Reference Table

| | PPO | GRPO | DPO |
|---|---|---|---|
| **Objective** | $\mathbb{E}[\min(r A, \text{clip}(r) A)]$ | $\mathbb{E}[\min(r A, \text{clip}(r) A)] - \beta \text{KL}$ | $-\mathbb{E}[\log\sigma(\beta \Delta)]$ |
| **Advantage** | $A = G - V(s)$ (GAE) | $A = (r - \mu) / \sigma$ (within-group) | Implicit (log-ratio) |
| **KL handling** | Fold into reward or loss | Add directly to loss | Implicit in loss structure |
| **Needs rollout** | ✓ | ✓ | ✗ |
| **Needs value head** | ✓ | ✗ | ✗ |
| **Needs reference** | ✓ | ✓ | ✓ |
| **Data source** | Online sampling | Online sampling (G per prompt) | Offline preference pairs |
| **Key hyperparams** | $\epsilon$, $\gamma$, `vf_coef` | $\epsilon$, $G$, $\beta$ | $\beta$ |

---

## References

1. Lambert, N. "Reinforcement Learning from Human Feedback." (2024)
2. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
3. Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023
4. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300 (2024)
5. Ouyang, L., et al. "Training language models to follow instructions with human feedback." NeurIPS 2022
