---
author: Jing Lu
pubDatetime: 2025-09-07T00:00:00Z
title: "User Interest Modeling with Transformer Architectures"
featured: false
draft: false
tags:
  - ML
  - Transformer
  - RecSys
  - Embeddings
description: "Exploring position embeddings, architecture choices, and training techniques for Transformer-based recommender systems."
---

## Position Embedding in Transformer-Based Recommender Systems

Early versions of the Transformer employ fixed sinusoidal positional encodings, giving each token position a unique identity while ensuring that nearby positions receive similar vectors so the model can infer distance.

Subsequent work on large-language models popularized relative position encodings such as Rotary Positional Embedding (RoPE). RoPE expresses positions as rotations in complex space, capturing relative offsets and scaling gracefully to very long contexts.

In most Transformer-based recommender-system papers, positional information is still supplied through a *learned* embedding lookup. This introduces extra parameters and allows the model to learn position-specific biases, but it may fail to generalize to positions that are rarely or never observed during training.

### RoPE Experiments

To assess RoPE's effectiveness, an experiment was conducted to replace the learned positional embeddings with the parameter-free RoPE formulation. This removes the parameters dedicated to position lookup while preserving overall model capacity.

Initially, the RoPE version demonstrated superior performance for the first 25,000 steps. However, after this point, the baseline version with learned embeddings exhibited a sharp increase in performance. This discrepancy might be attributable to the RoPE version potentially lacking sufficient parameters to capture as much information as the baseline.

An architecture combining RoPE and learned embeddings was also tested. This model learned slower, but its validation recall@10 converged when trained for longer durations.

---

## Model Architecture

Modern Transformer architectures commonly adopt several architectural choices that are predominantly used for natural language modeling. These may not be optimal for recommendation tasks. Key areas of experimentation include:

- **Normalization:** Using Layer Normalization before the non-residual parts of the network, and evaluating RMSNorm as a replacement for LayerNorm.
- **Feed-Forward Networks (FFN):** Dropping the bias term for more stable optimization and using gated MLP variants like GeGLU or SwiGLU instead of GeLU.
- **Dimensionality:** Adjusting the feed-forward dimension (`d_ff`) relative to the model dimension (`d_model`) and tuning the aspect ratio.
- **Component Arrangement:** Experimenting with serial versus parallel arrangements of the MLP and attention blocks.

### RMSNorm vs. LayerNorm

RMSNorm presents a more efficient alternative to LayerNorm by omitting mean calculation and bias terms, which reduces both operations and parameters. Although normalization accounts for a small fraction of the total FLOPs, it can be a significant portion of the runtime (around 25% in some cases) due to data movement.

In experiments, the model using RMSNorm performed slightly worse (a 0.7% relative decrease in recall) but was significantly faster, reducing training time by 25%.

### SwiGLU vs. GeLU

Gated MLP variants like SwiGLU can enhance a layer's expressive power, improve gradient flow, and allocate more capacity to efficient matrix-multiplication operations. By using a gated variant while keeping the number of parameters constant, a **1.2% improvement in validation Recall@10** was achieved.

---

## Training Techniques and Regularization

### L2 Regularization on Embeddings

The L2 norm of the embeddings can increase rapidly during training, even after recall and validation loss have plateaued. When the embedding vectors become too large, numerical precision can be affected.

```python
loss += lambda * torch.norm(user_values, p=2, dim=-1).mean()
```

For more precise control, a hinge L2 penalty can be used:

```python
user_norm = torch.norm(user_values, p=2, dim=-1).mean()
loss += lambda * torch.clamp(user_norm - self.l2_cap_tau, min=0.0) ** 2
```

| Configuration | val_recall@10 | L2 Norm |
|---------------|---------------|---------|
| Baseline (no penalty) | 0.9152 | 338.6 |
| L2 penalty = 0.0001 | 0.9138 | 281.2 |
| L2 penalty = 0.0005 | 0.9034 | 166.5 |
| Hinge (τ=180, λ=0.001) | 0.9114 | 171.5 |

**Recommendation:** A non-hinge penalty with `l2_penalty=0.0001` matches baseline recall while curbing norm growth.

---

## Feature Engineering

### Improved Temporal Features

Temporal information is critical for user interest modeling. A common approach is absolute timestamp encoding, but relative time signals can also be highly informative.

**HybridTimestampEncoding**: Half of the embedding dimensions encode the absolute timestamp, while the other half encode relative time information.

Results:
- Validation loss: 4.6956 (vs 4.6960 baseline) — 0.009% improvement
- Recall@10: 0.6998 (vs 0.6975 baseline) — 0.3% improvement

While the gains are modest, they are consistent—relative timing information provides reliable improvement.

