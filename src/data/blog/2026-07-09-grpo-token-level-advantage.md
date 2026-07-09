---
author: Jing Lu
pubDatetime: 2026-07-09T18:00:00Z
title: "From GRPO Outcome Rewards to Token-Level Advantage"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Reinforcement Learning
  - RLHF
  - Post Training
description: "A practical framework for turning GRPO-style sequence rewards into token-level advantages, including GAE-style estimators, credit assignment routes, and multi-reward training design."
---

GRPO works because it makes outcome-based reinforcement learning cheap.

For a prompt, sample a group of responses. Score each response with an outcome reward. Normalize those rewards inside the group. Use the resulting relative score as the advantage. Update the policy without training a separate critic.

That is a powerful simplification. It is also a coarse one.

In the usual outcome-supervised GRPO setup, every token in the same response receives the same advantage. A correct response gives positive pressure to the whole trajectory. An incorrect response gives negative pressure to the whole trajectory. This is often enough to improve pass@1 on math and code tasks, but it does not answer a more precise question:

```text
Which tokens should actually receive credit?
```

Long reasoning traces make this question unavoidable. Some tokens are real decision points. Some are syntax. Some are harmless filler. Some are actively bad side effects: repetition, over-excitement, verbosity, formatting drift, or stylistic artifacts that should not be reinforced just because the final answer happened to be correct.

The goal of this post is to make the token-level version explicit:

```text
Given a GRPO-style sequence reward, how do we construct token-level advantages?
```

And then the harder practical question:

```text
What happens when the training signal is not one reward, but a mixture of correctness,
format, fluency, KL, length, safety, process quality, and style?
```

The short answer is:

1. A single scalar sequence reward does not uniquely identify token-level credit.
2. GAE can turn rewards and values into token-level advantages, but it does not magically solve credit assignment.
3. The minimal engineering path is terminal GRPO reward plus a value head plus GAE.
4. Better credit assignment needs extra structure: traces, entropy, prefix values, tree structure, suffix resampling, reward redistribution, token reward models, or counterfactuals.
5. When multiple rewards are involved, they should be projected to tokens separately and only then combined into a final token-level advantage.

## What GRPO Gives Each Token

Let a policy generate a response \(y_{1:T}\) for prompt \(x\). In an outcome reward setting, the reward is usually observed only after the full response:

\[
R = R(x, y_{1:T})
\]

For math, \(R\) may be correctness. For code, it may be test pass rate. For RLHF, it may be a reward model score. For safety or formatting, it may be a judge score.

GRPO samples a group of \(G\) responses for the same prompt:

\[
\{y^{(i)}\}_{i=1}^{G}
\]

Each response receives a scalar reward \(R_i\). GRPO then normalizes those rewards within the group:

\[
Z_i = \frac{R_i - \mu_G}{\sigma_G + \epsilon}
\]

where:

\[
\mu_G = \frac{1}{G}\sum_{j=1}^{G} R_j
\]

The outcome-GRPO approximation is:

\[
A^{\text{GRPO}}_{i,t} = Z_i,\qquad t=1,\dots,T_i
\]

Every token in the response receives the same advantage.

This is not wrong as a policy-gradient estimator. The full trajectory received the reward, and every token action participated in that trajectory. But it is weak as credit assignment. It treats the decisive algebra step, the final boxed answer, a comma, and a repeated filler phrase as if they carried the same learning signal.

This is why recent GRPO variants and RLVR credit-assignment papers focus on length bias, token aggregation, entropy-aware weighting, eligibility traces, prefix values, and counterfactual continuations.

Useful starting points:

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Understanding R1-Zero-Like Training / Dr. GRPO](https://arxiv.org/abs/2503.20783)
- [GRPO-\(\lambda\)](https://arxiv.org/abs/2510.00194)

## Why a Sequence Reward Cannot Be Inverted

Suppose a response has length \(T\), and the final reward is \(R=1\). We want token rewards:

\[
r_1, r_2, \dots, r_T
\]

such that:

\[
\sum_{t=1}^{T} r_t = R
\]

There are infinitely many decompositions.

Uniform:

\[
r_t = \frac{1}{T}
\]

Terminal:

\[
r_T = 1,\qquad r_{t<T}=0
\]

Key-token:

\[
r_t = 0 \text{ except near the decisive reasoning step}
\]

The final scalar reward alone cannot tell us which decomposition is correct. A token-level advantage estimator must therefore introduce an additional assumption.

Common assumptions include:

- Position: later tokens are closer to the answer.
- Entropy: high-entropy tokens are real decision points.
- Prefix value: if a prefix sharply changes expected success, tokens near that change deserve credit.
- Counterfactual contribution: if replacing a token or suffix changes reward, the original token had credit.
- Process supervision: intermediate steps can be judged locally.
- Reward-model attribution: a sequence reward model can be decomposed into token or span contributions.

GAE does not remove the need for these assumptions. GAE answers a different question: given rewards and values, how do we estimate advantage with a useful bias-variance tradeoff?

The GAE residual is:

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

and the advantage estimate is:

\[
\hat A_t^{(\gamma,\lambda)}
=
\sum_{l=0}^{T-t}(\gamma\lambda)^l \delta_{t+l}
\]

If \(r_t\) is only terminal, GAE can still produce token-level advantages. But token differences mostly come from the value function, not from localized reward evidence.

Related foundations:

- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## The Minimal Path: Terminal GRPO Reward Plus GAE

The smallest useful change to GRPO is:

1. Keep the group-normalized score \(Z_i\).
2. Treat \(Z_i\) as a terminal reward.
3. Add a lightweight value head to the policy model.
4. Compute GAE over response tokens.

For response \(i\):

\[
r_{i,t} =
\begin{cases}
0, & t < T_i \\
Z_i, & t = T_i
\end{cases}
\]

Then:

\[
\delta_{i,t}
=
r_{i,t}
+ \gamma V_\psi(s_{i,t+1})
- V_\psi(s_{i,t})
\]

\[
\hat A_{i,t}
=
\delta_{i,t}
+ \gamma\lambda \hat A_{i,t+1}
\]

A minimal implementation looks like this:

```python
for prompt in batch:
    responses = sample_group(policy_old, prompt, G)
    rewards = [outcome_reward(prompt, y) for y in responses]
    z_scores = normalize_within_group(rewards)

    for y_i, z_i in zip(responses, z_scores):
        logits, values = policy_with_value_head(prompt, y_i)
        T = len(y_i)

        rewards_t = zeros(T)
        rewards_t[T - 1] = z_i

        advantages = zeros(T)
        last_gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            delta = rewards_t[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
            next_value = values[t]

        policy_loss += clipped_policy_loss(logits, y_i, advantages)
        value_loss += mse(values, returns_from(rewards_t, gamma))
```

This gives a real token-level advantage estimate. It preserves the GRPO sampling pipeline and only adds a value head.

The weakness is equally important: this is token-level bootstrapping, not necessarily faithful token-level credit assignment. Long-CoT PPO failures are often value-estimation failures: value bias, reward signal decay, and heterogeneous sequence lengths. VC-PPO and VAPO are examples of work that tries to make value-based methods reliable again for long reasoning.

Related work:

- [What's Behind PPO's Collapse in Long-CoT?](https://arxiv.org/abs/2503.01491)
- [VAPO](https://arxiv.org/abs/2504.05118)

## Routes to Better Token Credit

Once the minimal GAE version exists, the real design choice is how to define token rewards or token weights.

We can write the problem as a credit operator:

\[
\mathcal C: Z_i \mapsto \{r_{i,t}\}_{t=1}^{T_i}
\]

After that, GAE is standard:

\[
\hat A_{i,t} = \text{GAE}(r_{i,t}, V(s_{i,t}))
\]

### 1. Token Weighting: Position, Entropy, and Eligibility Traces

The cheapest route is weighted redistribution:

\[
r_{i,t}
=
Z_i \cdot \frac{w_{i,t}}{\sum_k w_{i,k}}
\]

The weights can come from:

- uniform token weights
- recency weighting
- early-and-late weighting
- token entropy
- step boundaries
- eligibility traces
- high-entropy masks

This family includes much of the lightweight GRPO-credit work. GRPO-\(\lambda\) applies trace-style token weighting. S-trace uses selective eligibility traces and keeps high-entropy tokens. HAPO frames hindsight credit through reward polarity and token entropy.

The advantage is cost. The disadvantage is that the weighting rule is still a hypothesis, not direct evidence.

Related work:

- [GRPO-\(\lambda\)](https://arxiv.org/abs/2510.00194)
- [S-trace](https://arxiv.org/html/2605.05965v1)
- [Where Hindsight Credit Can Reside / HAPO](https://arxiv.org/abs/2604.11056)
- [GTPO and GRPO-S](https://openreview.net/forum?id=CFF6zXErgS)

### 2. Prefix Values and Prefix Trees

A more structured route estimates the value of prefixes:

\[
U(s_t) \approx \mathbb E[Z \mid s_t]
\]

Then redistributed reward can be:

\[
r_t = U(s_t) - U(s_{t-1})
\]

This asks: after seeing this token, did the expected final outcome change?

RUDDER uses this idea in delayed-reward RL: learn a return decomposition and redistribute reward toward the events that explain future return. TEMPO/P2T adapts a related idea to grouped LLM rollouts. It turns multiple responses to the same prompt into a prefix tree, estimates prefix values from descendant outcomes, and adds temporal-difference corrections at branching tokens. Non-branching tokens reduce toward ordinary GRPO.

This is attractive because GRPO already samples groups for each prompt. The group is not just a variance-reduction device; it is a small local search tree.

Related work:

- [RUDDER](https://arxiv.org/abs/1806.07857)
- [TEMPO / Prefix-to-Tree](https://arxiv.org/abs/2509.18314)

### 3. Suffix Resampling and Counterfactual Continuations

Another route is to resample from an intermediate prefix.

For a position \(t\), keep the prefix \(y_{<t}\), sample suffixes:

\[
\tilde y_{t:T}^{(m)} \sim \pi(\cdot \mid x, y_{<t})
\]

and estimate:

\[
V(s_t)
\approx
\frac{1}{M}\sum_{m=1}^{M}
R(x, y_{<t}, \tilde y_{t:T}^{(m)})
\]

This gives a Monte Carlo prefix value. VinePPO uses this kind of language-environment flexibility to produce better credit estimates than learned value networks in reasoning tasks. Reset methods go further: identify or sample an intermediate reasoning state, reset there, generate counterfactual suffixes, and learn from outcome differences.

The cost is obvious. You cannot afford to do this for every token in every response. The practical version should target high-entropy tokens, step boundaries, suspicious reasoning steps, or spans selected by a cheaper heuristic.

Related work:

- [VinePPO](https://proceedings.mlr.press/v267/kazemnejad25a.html)
- [Credit Assignment with Resets in Language Model Reasoning](https://arxiv.org/abs/2605.25507)

### 4. Learned Token Reward Models

Instead of inferring credit online, train a token reward model:

\[
f_\phi(x, y_{\le t}) \to \phi_t
\]

Then project its token scores back to the known sequence score:

\[
r_t
=
\phi_t
+
\frac{Z - \sum_k \phi_k}{T}
\]

This enforces:

\[
\sum_t r_t = Z
\]

TLCR trains a discriminator to produce continuous token-level reward. Q-RM decouples reward modeling from generation and learns token-level Q-function rewards from preference data. R3HF treats reward redistribution as a way to decompose a sequence reward model's output into token-level contributions.

This route is especially useful for local quality signals: fluency, style, repetition, format, and harmlessness. It is also a good way to distill expensive counterfactual or Shapley-style credit into a cheap online model.

Related work:

- [TLCR](https://arxiv.org/abs/2407.16574)
- [Q-RM / Discriminative Policy Optimization for Token-Level Reward Models](https://arxiv.org/abs/2505.23363)
- [R3HF](https://arxiv.org/abs/2411.08302)

### 5. Token Aggregation Instead of Explicit Reward Redistribution

Some methods do not explicitly produce token rewards. They instead change how token losses are aggregated.

DAPO changes GRPO's sample-level averaging into token-level policy-gradient loss and adds training-stability improvements. Dr. GRPO corrects token aggregation bias. TEPO links group-level reward to token aggregation through sequence-level likelihood and adds token-level KL masking.

This family is simple to integrate with existing GRPO pipelines. It is better understood as token weighting or aggregation control rather than a fully interpretable token reward decomposition.

Related work:

- [DAPO](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO](https://arxiv.org/abs/2503.20783)
- [TEPO](https://arxiv.org/abs/2604.12736)

## Multi-Reward Training: Split First, Combine Later

Real training almost never has one reward.

A reasoning model may receive:

- correctness reward
- process reward
- format reward
- fluency reward
- repetition penalty
- over-excitement penalty
- length penalty
- KL penalty
- safety reward
- tool-use or execution reward

The tempting implementation is:

\[
R_{\text{total}}
=
R_{\text{task}}
+ \beta_1 R_{\text{format}}
+ \beta_2 R_{\text{style}}
- \beta_3 R_{\text{KL}}
\]

and then broadcast or redistribute \(R_{\text{total}}\).

That is usually the wrong abstraction.

Imagine a math response is correct, but contains:

```text
Obviously!!! This is super easy!!!
```

If correctness and style are collapsed into one sequence reward, the model can learn either bad lesson:

1. Because the answer was correct, the over-excited phrase is reinforced.
2. Because the style was bad, the useful reasoning steps are penalized.

The fix is to split reward channels before token credit assignment:

\[
A_{t}^{\text{final}}
=
A_{t}^{\text{task}}
+ \beta_{\text{format}} A_{t}^{\text{format}}
+ \beta_{\text{style}} A_{t}^{\text{style}}
+ \beta_{\text{process}} A_{t}^{\text{process}}
- \beta_{\text{KL}} k_t
\]

Each reward channel should have its own projection to token level.

### Reward Channels Have Different Natural Granularities

Not every reward should be distributed the same way.

| Reward type | Natural granularity | Recommended projection |
|---|---:|---|
| Final correctness | sequence / terminal | GRPO-\(\lambda\), prefix tree, suffix resampling, terminal GAE |
| Unit-test pass rate | sequence / test case | code span, function, line, or suffix resampling |
| Process correctness | step | step reward distributed across step tokens |
| Format | token / span | local penalty on malformed spans |
| Fluency | span | token or span reward model |
| Over-excitement | span | local penalty on exaggerated phrases and punctuation |
| Repetition | n-gram / span | local penalty on repeated spans |
| Length | sequence plus token | small per-token cost plus overlong penalty |
| KL | token | direct per-token KL |
| Safety | span / sequence | local penalty for explicit violation, hard constraint for severe cases |

The principle is simple:

```text
Assign reward where the evidence occurs.
```

Do not broadcast a local style violation across the reasoning trace. Do not reward every filler token because the final answer is correct. Do not turn per-token KL into a sequence scalar unless you are willing to lose its local control.

### Channel-Wise Normalization

Reward channels have different scales and variances. Correctness may be binary. KL is measured per token. Style may be sparse and negative. Format may be almost always zero except on invalid outputs.

If all channels are mixed first and whitened later, a high-variance channel can dominate the others.

A safer pattern is:

\[
\tilde A_{i,t}^{(c)}
=
\text{normalize within channel } c
\]

then:

\[
A_{i,t}^{\text{final}}
=
\sum_c \beta_c \tilde A_{i,t}^{(c)}
\]

If the coefficients \(\beta_c\) change over training, use multiple value heads:

```text
shared transformer
  |-- policy head
  |-- V_task head
  |-- V_process head
  |-- V_style head
  |-- V_format head
```

A single value head trained on a changing weighted mixture can become poorly calibrated. Separate heads let each channel estimate its own return, and the final policy advantage can be assembled after normalization.

### Constraints Are Often Better Than Fixed Weights

For auxiliary quality signals, fixed scalar weights are brittle.

Too much style penalty can suppress reasoning. Too little does nothing. Too much length penalty creates short but shallow answers. Too much KL prevents exploration.

For some channels, a constrained objective is cleaner:

\[
\max_\pi \mathbb E[R_{\text{task}}]
\]

subject to:

\[
\mathbb E[\text{style violation}] \le \tau_{\text{style}}
\]

\[
\mathbb E[\text{format error}] \le \tau_{\text{format}}
\]

In practice:

```python
beta_style = max(0.0, beta_style + eta * (style_rate - target_style_rate))
beta_format = max(0.0, beta_format + eta * (format_error_rate - target_format_rate))
```

This allows the model to focus on task learning when style is under control, and only tighten style pressure when violations rise.

Safety and tool-call validity are often better treated as hard or lexicographic constraints:

```text
First satisfy safety and format.
Then maximize task reward.
Then optimize fluency, length, and style.
```

## A Practical Implementation Plan

If I were adding token-level advantage to an existing GRPO trainer, I would stage it this way.

### V0: Fix Loss Aggregation Bias

Implement DAPO/Dr. GRPO-style token-level aggregation so length normalization does not create obvious per-token bias.

This does not solve credit assignment, but it makes the baseline healthier.

### V1: Add Terminal-GRPO GAE

Use group-normalized \(Z_i\) as terminal reward:

\[
r^{\text{task}}_{i,T_i} = Z_i
\]

Add a shared value head and compute:

\[
A^{\text{task}}_{i,t}
=
\text{GAE}(r^{\text{task}}, V^{\text{task}})
\]

This is the minimal token-level advantage.

### V2: Make Style, Format, and KL Token-Local

Do not fold these into the task reward.

Use local rewards:

```python
r_style[t] = 0.0

if span_is_over_excited(t):
    r_style[t] -= 1.0

if span_is_repetitive(t):
    r_style[t] -= 1.0

if span_is_ungrammatical(t):
    r_style[t] -= 0.5

r_kl[t] = -(logp_policy[t] - logp_ref[t])
```

Then combine:

\[
A_t
=
A_t^{\text{task}}
+ \beta_{\text{style}} A_t^{\text{style}}
- \beta_{\text{KL}} k_t
\]

The important part is locality: a bad phrase should not penalize the whole proof.

### V3: Improve Task Credit

Once the simple version is stable, improve the task channel:

- Add GRPO-\(\lambda\) or S-trace weighting.
- Use entropy masks to focus on decision tokens.
- Build prefix trees from the sampled group and add branch-level TD corrections.
- Try prefix value heads or RUDDER-style redistribution.

This step improves correctness credit without mixing it with style control.

### V4: Use Counterfactuals as a Teacher

Run expensive suffix resampling only on selected positions:

```python
for step in selected_high_entropy_or_suspicious_steps:
    suffixes = resample_suffixes(prompt, prefix, M)
    prefix_value[step] = mean(outcome_reward(s) for s in suffixes)
```

Use those high-quality estimates to train a cheaper token reward or prefix value model.

Counterfactual credit is too expensive to apply everywhere, but it can be an excellent teacher.

## A Unified Estimator

The estimator I would actually aim for is:

\[
A^{\text{final}}_{i,t}
=
\alpha_{\text{task}} \tilde A^{\text{task}}_{i,t}
+ \alpha_{\text{process}} \tilde A^{\text{process}}_{i,t}
+ \alpha_{\text{style}} \tilde A^{\text{style}}_{i,t}
+ \alpha_{\text{format}} \tilde A^{\text{format}}_{i,t}
- \alpha_{\text{KL}} \tilde k_{i,t}
- \alpha_{\text{length}} \tilde \ell_{i,t}
\]

Where:

- \(\tilde A^{\text{task}}\) comes from GRPO group reward plus terminal GAE, GRPO-\(\lambda\), TEMPO, or suffix resampling.
- \(\tilde A^{\text{process}}\) comes from step verification or process reward.
- \(\tilde A^{\text{style}}\) comes from local style rules or a token reward model.
- \(\tilde A^{\text{format}}\) comes from validators and only affects malformed spans.
- \(\tilde k_t\) is per-token KL.
- \(\tilde \ell_t\) captures length and repetition pressure.

Every component is normalized in its own channel before the final weighted sum.

This keeps reward semantics clear:

```text
Task reward decides whether the model solved the problem.
Process reward decides whether intermediate reasoning steps are good.
Style reward decides how the answer is expressed.
Format reward enforces structure.
KL keeps the policy from drifting too far.
Length and repetition rewards prevent waste.
```

The channels meet at the advantage layer, not at the raw sequence-reward layer.

## What to Measure

Final task score is not enough. A token-level advantage estimator should be evaluated on both performance and credit behavior.

I would track:

| Category | Metrics |
|---|---|
| Task quality | math accuracy, code pass@1, reward model score, judge win-rate |
| Training efficiency | reward per rollout, steps to threshold, token budget to threshold |
| Stability | policy KL, token entropy, advantage variance, value explained variance |
| Output quality | response length, repetition rate, overlong rate, format error rate |
| Attribution faithfulness | reward drop after deleting top-credit tokens vs random tokens |
| Reward interference | correctness drop after style reward, exploration drop after KL, short-answer bias after length penalty |

Synthetic oracle tasks are especially useful. Create tasks where only a few known tokens or steps determine reward. Then measure whether the estimator actually identifies those tokens. This is much cleaner than trying to infer credit quality from benchmark accuracy alone.

## A Small Synthetic Check

Before treating this as a training recipe, I ran a small synthetic credit-assignment check.

The setup is intentionally simple. Each generated sequence has 48 tokens. Four hidden tokens carry the true task credit. Some sequences also contain a local style violation span, meant to mimic things like over-excitement, repeated punctuation, or repetitive filler. The estimator sees only sequence-level rewards plus cheap token observables: token entropy, a noisy decision signal, and local style flags. Because this is synthetic, we know the hidden token-level credit and can directly measure whether an estimator puts credit in the right places.

The experiment used 20 random seeds, 500 prompts per seed, and 8 responses per prompt. The metrics below are averaged across seeds.

| Estimator | Token-credit correlation | Task key recall | Style violation recall | Filler credit mass |
|---|---:|---:|---:|---:|
| Task uniform | 0.140 | 0.075 | 0.003 | 0.903 |
| Task terminal trace | 0.102 | 0.000 | 0.008 | 0.908 |
| Task entropy weighting | 0.489 | 1.000 | 0.000 | 0.569 |
| Task learned token RM | 0.965 | 0.926 | 0.021 | 0.316 |
| Mixed total entropy weighting | 0.457 | 1.000 | 0.000 | 0.569 |
| Split task entropy + local style | 0.502 | 0.839 | 1.000 | 0.491 |
| Split learned task + local style | 0.966 | 0.837 | 0.560 | 0.295 |

The numbers are not meant to prove a universal ranking of methods. The environment was designed to isolate a specific failure mode.

Uniform and terminal-trace redistribution mostly put credit on filler tokens. This is expected: they carry the sequence sign, but they do not localize causality.

Entropy weighting finds the task-critical tokens perfectly in this toy setup, because the hidden decision tokens were constructed to be high entropy. But when task reward and style reward are mixed into one total reward, entropy weighting still misses the style violation spans entirely. It knows where the model made uncertain task decisions; it does not know where a local style problem occurred.

The split-channel estimator fixes that failure. It projects task reward through task-oriented token credit, applies style penalty locally, and only then combines the channels. In the synthetic mixed-reward setting, this gives near-perfect style violation recall while preserving much of the task-key localization.

The learned token reward model does best when a noisy local decision feature is available. That is the encouraging part: even sequence-level labels can train useful token-level reward structure if the model has token-local features that correlate with hidden credit.

The practical lesson is the same as the design rule above:

```text
Do not mix rewards first and ask token credit assignment to untangle them later.
Project each reward channel at its natural granularity, then combine advantages.
```

## The Main Takeaway

GRPO showed that outcome rewards can train reasoning models without a heavy critic. But outcome broadcast is not the end of the story. It is a useful first approximation.

Moving from sequence reward to token-level advantage requires choosing a credit assumption:

```text
terminal reward + value head
trace weighting
entropy-aware weighting
prefix values
prefix trees
suffix resampling
reward redistribution
token reward models
counterfactual attribution
```

The right route depends on the constraint:

- If you need the smallest engineering change, use terminal GRPO reward plus GAE.
- If you need cheap improvement, use GRPO-\(\lambda\), S-trace, or entropy-aware weighting.
- If you want to exploit group rollout structure, use prefix trees.
- If you can train auxiliary models, use reward redistribution or token reward models.
- If you need high-quality credit labels, use counterfactuals or Shapley-style methods as a teacher.

For multi-reward training, the rule is even more important:

```text
Split reward channels first.
Project each channel to token level.
Normalize within channel.
Combine at the advantage layer.
```

Correctness should teach the model to solve the problem. Style should teach it how to speak. Format should enforce structure. KL should prevent drift. Length and repetition should control waste. These signals should cooperate, not overwrite each other.

That is the natural next step after GRPO: keep the group-relative efficiency, but add token-level credit and multi-channel reward accounting.

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- [What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret](https://arxiv.org/abs/2503.01491)
- [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- [GRPO-\(\lambda\): Credit Assignment improves LLM Reasoning](https://arxiv.org/abs/2510.00194)
- [KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning](https://arxiv.org/abs/2505.16826)
- [Exploiting Tree Structure for Credit Assignment in RL Training of LLMs](https://arxiv.org/abs/2509.18314)
- [VinePPO: Refining Credit Assignment in RL Training of LLMs](https://proceedings.mlr.press/v267/kazemnejad25a.html)
- [Beyond Uniform Credit Assignment: Selective Eligibility Traces for RLVR](https://arxiv.org/html/2605.05965v1)
- [Where Hindsight Credit Can Reside: A Signed-Capacity View of Token Updates in RLVR](https://arxiv.org/abs/2604.11056)
- [Token-Level Policy Optimization via Sequence-Level Likelihood](https://arxiv.org/abs/2604.12736)
- [Credit Assignment with Resets in Language Model Reasoning](https://arxiv.org/abs/2605.25507)
- [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857)
- [R3HF: Reward Redistribution for Enhancing Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2411.08302)
- [TLCR: Token-Level Continuous Reward for Fine-grained RLHF](https://arxiv.org/abs/2407.16574)
- [Discriminative Policy Optimization for Token-Level Reward Models](https://arxiv.org/abs/2505.23363)
