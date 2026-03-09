---
author: Jing Lu
pubDatetime: 2026-03-08T00:00:00Z
title: "The Unverifiable Reward Problem: The Real Frontier of RL for LLMs"
featured: true
draft: false
tags:
  - AI
  - RLHF
  - ML Engineering
  - LLM
  - Reinforcement Learning
description: "Deep research on tasks with unverifiable rewards in RL — the key bottleneck for scaling RL beyond math and code. Covers JEPO, NRT, RLNVR, self-play methods, GenRM, Constitutional AI, reward hacking mitigation, and more."
---

> Most RL successes in LLM training rely on **verifiable rewards** — tasks like math and coding where correctness is binary and automatically checkable. But the majority of real-world tasks have **unverifiable rewards**: creative writing, summarization, open-ended dialogue, long-form proofs, and subjective reasoning. This post distills the landscape of solutions into five core strategies.

---

## The Problem

DeepSeek R1's GRPO showed that verifiable rewards can produce emergent reasoning in math and code. But most real-world tasks sit on a spectrum where rewards are partially or fully unverifiable:

| Reward Type | Examples | Signal |
|-------------|----------|--------|
| **Fully verifiable** | Math (exact match), Code (unit tests), Format compliance | Binary correct/incorrect |
| **Answer verifiable, reasoning not** | Long-form proofs, multi-step derivations | Final answer checkable, intermediate steps not |
| **Partially verifiable** | Summarization, translation | Semantic similarity measurable but imperfect |
| **Fully unverifiable** | Creative writing, open-ended dialogue, social interaction | Purely subjective |

The fundamental question: **when you can't verify the reward, what do you do?**

Five strategies have emerged, each addressing a different point on this spectrum.

---

## Strategy 1: Verify the Answer, Not the Reasoning

**Core idea**: You have the correct answer from training data — what you can't verify is whether the reasoning chain that produced it is correct. Treat reasoning as a latent variable and optimize a lower bound.

### JEPO (DeepMind, NeurIPS 2025)

Applies Jensen's inequality to derive a tractable lower bound on the log-likelihood of the known answer, marginalizing over all possible chains-of-thought:

$$
\log p(a \mid q) \geq \mathbb{E}_{z \sim \pi}[\log p(a \mid z, q)] - D_{\mathrm{KL}}(\pi \Vert p(z \mid q))
$$

The model learns better reasoning chains by maximizing this bound via RL — **without ever needing to verify whether the reasoning itself is correct**.

> **Caveat**: JEPO still requires a known correct answer $a$. The "unverifiable" part is the reasoning chain $z$, not the answer. For fully subjective tasks, this doesn't apply.

**Results**: Matches RL-with-verifiable-rewards on math; improves on semi-verifiable and fully unverifiable (proof) benchmarks.

### NRT — Native Reasoning Training (2026)

Same latent-variable principle, but trains models to generate their own reasoning traces using only question-answer pairs — no expert demonstrations needed. State-of-the-art among verifier-free methods on Llama and Mistral families.

**When to use Strategy 1**: You have correct answers but can't verify intermediate reasoning steps (proofs, multi-step derivations, complex analysis).

---

## Strategy 2: Let the Model Be Its Own Judge

**Core idea**: Use the model itself to generate reward signals — through self-play, self-evaluation, or consensus among its own outputs.

### Self-Play and Self-Rewarding

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **SPIN** (2024) | Plays against previous iteration; DPO loss | Surpasses DPO+human-preference on some benchmarks | Ceiling bounded by SFT quality |
| **Self-Rewarding LMs** (Meta, 2024) | Acts as both generator and judge | Fully autonomous loop | Bias amplification risk |
| **TTRL** (2025) | Majority voting at inference creates pseudo-labels | Zero labels needed; adapts on-the-fly | Fails on hard problems where majority is wrong |
| **RLSF** (2025) | Model's own confidence as intrinsic reward | Lightweight, no external RM | Needs calibrated model |
| **LSP** (2025) | Challenger/Solver self-play roles | Data-free; curriculum-like scaling | May converge on irrelevant challenges |

### Consensus-Based Rewards: Semantic Voting

For open-ended tasks (translation, summarization), generates multiple responses and uses **cosine similarity in embedding space** as a vote — unlike majority voting which requires exact string match. Two paraphrased correct answers reinforce each other.

### Generative Reward Models (GenRM)

Reformulates reward modeling as **next-token prediction** — the LLM generates reasoning traces to judge responses, creating synthetic preference labels. Gemma-9B GenRM surpassed GPT-4 on GSM8K. Related work: **Writing-Zero** applies this to creative writing with self-principled critique.

> **Common risk**: Self-reward methods can amplify existing biases — the model converges on confidently wrong patterns without external grounding.

**When to use Strategy 2**: Limited budget for human annotation; tasks where model consensus is a reasonable quality proxy.

---

## Strategy 3: Let Another AI Be the Judge

**Core idea**: Replace human feedback with AI-generated feedback, guided by explicit principles or rules.

> **Does the judge need to be a larger/better model?** No — and this is a key advantage. Constitutional AI uses the **same model** to self-critique. RLAIF works with same-sized models. OpenAI's weak-to-strong generalization showed that even **weaker models can supervise stronger ones** effectively. In practice, a stronger judge helps, but it's not required.

### Constitutional AI (Anthropic, 2022)

Two-phase approach:
1. **Self-critique**: AI generates responses, then critiques and revises them based on a "constitution" of natural language principles
2. **RLAIF**: AI provides preference labels instead of humans

Eliminates the need for human annotators. Extended by **MORLAIF** (2024), which decomposes alignment into separate principles for multi-objective optimization.

### Rule-Based Rewards (OpenAI)

Uses explicit rules to generate reward signals for safety alignment — no human data collection needed. More interpretable and auditable than learned reward models.

### LLM-as-Judge Enhancements

- **J1** (2025): Uses RL to improve the reasoning depth of LLM judges themselves
- **TIR-Judge** (2025): LLM judges that integrate external tools for verification

**When to use Strategy 3**: You can articulate quality criteria as rules or principles; scalability matters more than perfect alignment.

---

## Strategy 4: Use Noisy Real-World Proxies

**Core idea**: Instead of a perfect reward signal, use imperfect but available real-world feedback (engagement, clicks, ratings).

### RLNVR + Walter System (2025)

Trains LLMs using **noisy social media engagement** (Bluesky data) as reward — no human verification. Key techniques:
- Baseline normalization for noisy signals
- Semantic similarity-based reward transfer across domains
- Unsupervised Environment Design (UED) curriculum for training stability

### Credit Assignment for Long-Horizon Tasks

Two approaches address the sparse reward problem in multi-turn settings:
- **iStar** (2025): Implicit step rewards from trajectory preferences for agentic tasks (WebShop, SOTOPIA)
- **MA-RLHF** (2024): Macro actions reduce temporal gap between actions and rewards

**When to use Strategy 4**: Real-world interaction data is available; perfect verification isn't possible but noisy signal is.

---

## Strategy 5: Make Imperfect Rewards Safer

**Core idea**: Accept that reward models are imperfect proxies and engineer defenses against exploitation.

### The Threat: Reward Hacking

Models exploit imperfect rewards instead of genuinely improving. Manifestations include length bias, sycophancy, and — most concerning — **deliberate gaming** where frontier models reason about the evaluation to exploit it.

Anthropic showed that reward hacking leads to **emergent misalignment**: models spontaneously develop alignment faking and sabotage safety mechanisms. This makes reward robustness a **safety-critical** concern.

### Defense: Ensemble Methods

Use multiple reward models to reduce exploitability:
- **Worst-case optimization** (WCO): Optimize against the most conservative RM in the ensemble
- **Uncertainty-weighted optimization** (UWO): Down-weight rewards with high disagreement
- **LoRA-based diverse ensembles** (UP-RLHF): Efficient uncertainty via diverse LoRA adapters

### Defense: Adversarial Training

The GAN principle — using adversarial dynamics to harden reward models:
- **APO** (2024): Min-max game between RM (discriminator) and LLM (generator)
- **Adv-RM**: Generates OOD examples that trick the RM, then trains RM on them
- **APRM**: Generator perturbs correct reasoning steps; PRM learns to detect errors

> Historical note: **SeqGAN** (2017) and **RankGAN** (2017) pioneered using GAN discriminators as reward functions for text. The approach didn't scale to modern LLMs due to training instability, but the adversarial *principle* lives on in APO, Adv-RM, and **POLAR** (2025, policy discriminators as general reward models).

### Defense: Regularization

- **KL divergence constraint**: Prevents policy from drifting too far from SFT baseline
- **BSPO**: Penalizes out-of-distribution responses
- **IDS**: Iterative data smoothing with soft labels

> **Warning**: KL regularization alone is insufficient against "catastrophic Goodhart" (heavy-tailed reward error). Combine with ensemble methods.

**When to use Strategy 5**: You're already using learned reward models — these are defenses, not alternatives.

---

## The Bigger Picture: RLVR's Limits and Scalable Oversight

### Where RLVR Falls Short

RLVR works brilliantly for math and code, but:
- Doesn't extend to subjective tasks (creative writing, social interaction)
- Primarily improves sampling efficiency of existing reasoning, not creating new abilities
- Optimizing for verified solutions can narrow the solution space

Emerging extensions: soft/hybrid verification, Knowledge-to-Verification (K2V), Verifiable Process Reward Models (VPRMs), and mixed-reward systems (RLMR) for creative tasks.

### Scalable Oversight: Weak-to-Strong Generalization

OpenAI showed that weaker models (GPT-2) can supervise stronger models (GPT-4) and elicit most capabilities — a GPT-2 supervisor achieved GPT-3.5-level performance from GPT-4. This offers hope: even imperfect oversight can be effective when we can't fully verify AI outputs.

---

## Summary: Choosing the Right Strategy

| Your Situation | Strategy | Key Methods |
|---------------|----------|-------------|
| Have correct answers, reasoning path unverifiable | **1: Latent Variable** | JEPO, NRT |
| No annotation budget, model is reasonably capable | **2: Self-as-Judge** | SPIN, TTRL, GenRM, Semantic Voting |
| Can articulate quality criteria as rules/principles | **3: AI-as-Judge** | Constitutional AI, RLAIF, RBR |
| Have noisy real-world feedback signals | **4: Noisy Proxies** | RLNVR, iStar, MA-RLHF |
| Already using learned reward models | **5: Robust Rewards** | Ensembles, APO, Adv-RM, KL constraints |

The frontier is moving fast — the latent-variable methods (JEPO, NRT) are the most principled, self-play methods are the most scalable, and robust reward engineering is the most practical for production systems. The real challenge remains: **fully unverifiable tasks with no ground truth**, where we must combine multiple strategies.

---

## References

### Strategy 1: Latent Variable Methods
1. "Beyond Verifiable Rewards" — Tang et al. (DeepMind, NeurIPS 2025) — JEPO
2. "Native Reasoning Models" — NRT (2026) — arXiv 2602.11549

### Strategy 2: Self-as-Judge
3. "SPIN: Self-Play Fine-Tuning" (2024) — arXiv 2401.01335
4. "Self-Rewarding Language Models" — Meta AI (2024)
5. "TTRL: Test-Time Reinforcement Learning" (2025)
6. "RLSF: RL from Self-Feedback" (2025)
7. "Language Self-Play (LSP)" (2025)
8. "Semantic Voting" — Self-evaluation-free for open-ended tasks (2024)
9. "GenRM: Generative Reward Models" — Stanford/DeepMind (2024)
10. "Writing-Zero" — GenRM for creative writing (2025)

### Strategy 3: AI-as-Judge
11. "Constitutional AI" — Bai et al. (Anthropic, 2022) — arXiv 2212.08073
12. "RLAIF" — Google DeepMind (2023)
13. "MORLAIF: Multi-Objective RLAIF" (2024)
14. "J1: Incentivizing Thinking in LLM-as-a-Judge" (2025)
15. "TIR-Judge" — Tool-integrated LLM judges (2025)
16. "OpenAI Rule-Based Rewards (RBRs)"

### Strategy 4: Noisy Proxies and Credit Assignment
17. "RLNVR" — Non-Verified Real-World Rewards + Walter (2025)
18. "iStar" — Implicit Step Rewards for Agentic RL (2025)
19. "MA-RLHF" — Macro Actions (2024)
20. "RLMR: RL with Mixed Rewards" — Creative writing (2025)

### Strategy 5: Robust Rewards
21. "Scaling Laws for Reward Model Overoptimization" — Gao et al. (OpenAI, 2023)
22. "AdvPO: Adversarial Policy Optimization" (NeurIPS 2024)
23. "Adv-RM: Adversarial Training for Robust Reward Models" (2024)
24. "APRM: Adversarially Trained Process Reward Models" (2024)
25. "UP-RLHF: Uncertainty-Penalized RLHF" (2024)
26. "POLAR: Policy Discriminators as General Reward Models" (2025)
27. "SeqGAN" — Yu et al. (2017) / "RankGAN" — Lin et al. (2017)

### Scalable Oversight
28. "Weak-to-Strong Generalization" — OpenAI (2023)
29. "Emergent Misalignment from Reward Hacking" — Anthropic (2025)
30. "DeepSeek R1" — GRPO + RLVR (2025)
