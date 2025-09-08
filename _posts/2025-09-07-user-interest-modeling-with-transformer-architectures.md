---
layout: post
title: "User Interest Modeling with Transformer Architectures"
---

# User Interest Modeling with Transformer Architectures

## Position Embedding in Transformer-Based Recommender Systems

Early versions of the Transformer employ fixed sinusoidal positional encodings, giving each token position a unique identity while ensuring that nearby positions receive similar vectors so the model can infer distance.

Subsequent work on large-language models popularized relative position encodings such as Rotary Positional Embedding (RoPE). RoPE expresses positions as rotations in complex space, capturing relative offsets and scaling gracefully to very long contexts.

In most Transformer-based recommender-system papers, positional information is still supplied through a *learned* embedding lookup. This introduces extra parameters and allows the model to learn position-specific biases, but it may fail to generalize to positions that are rarely or never observed during training.

Directly encoding a timestamp feature inside the positional embedding is uncommon in recommender systems because temporal signals are usually added as separate features. Absolute and relative timestamps at multiple granularities (e.g., hours since last interaction or day-of-week) already provide the model with rich temporal context, so the positional encoding need not carry this burden.

To assess RoPE’s effectiveness, an experiment was conducted to replace the learned positional embeddings with the parameter-free RoPE formulation. This removes the parameters dedicated to position lookup while preserving overall model capacity and delivering a more robust treatment of long-range dependencies.

Initially, the RoPE version demonstrated superior performance for the first 25,000 steps. However, after this point, the baseline version with learned embeddings exhibited a sharp increase in performance. Investigations into the scheduler's impact on this surge revealed minimal effect. This discrepancy might be attributable to the RoPE version potentially lacking sufficient parameters to capture as much information as the baseline.

An architecture combining RoPE and learned embeddings was also tested. This model learned slower, but its validation recall@10 converged when trained for longer durations.

# Model Architecture

Modern Transformer architectures commonly adopt several architectural choices that are predominantly used for natural language modeling. These may not be optimal for recommendation tasks. Key areas of experimentation include:

*   **Normalization:** Using Layer Normalization before the non-residual parts of the network, and evaluating RMSNorm as a replacement for LayerNorm.
*   **Feed-Forward Networks (FFN):** Dropping the bias term for more stable optimization and using gated MLP variants like GeGLU or SwiGLU instead of GeLU.
*   **Dimensionality:** Adjusting the feed-forward dimension (`d_ff`) relative to the model dimension (`d_model`) and tuning the aspect ratio of `d_model` to the number of layers (`n_layer`).
*   **Component Arrangement:** Experimenting with serial versus parallel arrangements of the MLP and attention blocks.

## RMSNorm vs. LayerNorm

RMSNorm presents a more efficient alternative to LayerNorm by omitting mean calculation and bias terms, which reduces both operations and parameters. Although normalization accounts for a small fraction of the total floating-point operations (FLOPs), it can be a significant portion of the runtime (around 25% in some cases) due to data movement. Therefore, the efficiency gains from RMSNorm can be substantial.

In experiments, the model using RMSNorm performed slightly worse (a 0.7% relative decrease in recall) but was significantly faster, reducing training time by 25%. However, when comparing wall-clock time against validation recall, the RMSNorm version was only marginally ahead of the LayerNorm version for a brief period during training.

## SwiGLU vs. GeLU

Gated MLP variants like SwiGLU can enhance a layer's expressive power, improve gradient flow, and allocate more capacity to efficient matrix-multiplication operations instead of dense activations. By using a gated variant while keeping the number of parameters constant, a 1.2% improvement in validation Recall@10 was achieved.

# Training Techniques and Regularization

## L2 Regularization on Embeddings

The L2 norm of the embeddings can increase rapidly during training, even after recall and validation loss have plateaued. When the embedding vectors become too large, numerical precision can be affected, and the optimizer may struggle. To mitigate this, an L2 penalty can be added to the loss function:

`loss += lambda * torch.norm(user_values, p=2, dim=-1).mean()`

The choice of the regularization strength `lambda` is critical. In practice, values greater than or equal to 0.01 can negatively impact recall, so this hyperparameter often needs to be retuned whenever the loss function is modified.

For more precise control over the embedding norms, a hinge L2 penalty can be used:

`user_norm = torch.norm(user_values, p=2, dim=-1).mean()`
`loss += lambda * torch.clamp(user_norm - self.l2_cap_tau, min=0.0) ** 2`

This penalizes deviations from a target norm `l2_cap_tau`, encouraging the embeddings to stay within a certain magnitude.

In an evaluation, a baseline model without any penalty achieved a `val_recall@10` of 0.9152, with the L2 norm of validation embeddings around 338.6. Introducing a small non-hinge L2 penalty of 0.0001 maintained a similar recall (0.9138) while reducing the norm by approximately 17% to 281.2. In contrast, larger penalties (e.g., 0.0005) led to a drop in recall to 0.9034, despite a more significant reduction in the norm (to 166.5). The hinge-capped L2 penalty provided stronger norm control with a modest trade-off in recall. The best-performing setting (with `tau=180` and `l2_penalty=0.001`) achieved a `recall@10` of 0.9114 with an embedding norm of approximately 171.5.

A non-hinge penalty with `l2_penalty=0.0001` is recommended to match the baseline recall while curbing norm growth. If stricter norm control is necessary and a small additional recall trade-off is acceptable, the hinge penalty with `tau=180` and `l2_penalty=0.001` is a viable alternative.

## Learning Rate Schedule, Warm-Up, and Weight Decay

A learning rate schedule with a warm-up period followed by a cosine decay was tested. This setup used a 5% warm-up phase, with the maximum number of steps aligned with the total training steps and a minimum learning rate of 1% of the initial learning rate. The AdamW optimizer was used with a `weight_decay` of 0.01. The validation metrics for this configuration were comparable to the baseline, and in some cases, slightly worse.

Another experiment involved a cosine learning rate schedule with warm-up and warm restarts. This approach showed strong performance early in the training process but eventually fell off, finishing below the baseline.

# Feature Engineering

## Improved Temporal Features

Temporal information is a critical component of user interest modeling. A common approach is to use an absolute timestamp encoding, which gives each interaction a unique temporal identifier. However, in many recommendation scenarios, relative time signals—such as the time elapsed between interactions—can also be highly informative.

To capture both aspects of time, a `HybridTimestampEncoding` was introduced. In this method, half of the embedding dimensions are dedicated to encoding the absolute timestamp, while the other half encode relative time information.

Experiments showed that:

*   The hybrid encoding resulted in a validation loss of 4.6956, a marginal but consistent 0.009% improvement over the 4.6960 achieved with the absolute-only encoding.
*   In terms of `Recall@10`, the hybrid approach achieved 0.6998, a 0.3% increase compared to the 0.6975 from the absolute-only encoding.

While the gains from incorporating relative temporal features are modest, they are consistent. This suggests that while the absolute timestamp captures the majority of the temporal signal, relative timing information still provides a reliable, albeit small, improvement.
