---
author: Jing Lu
pubDatetime: 2025-09-07T00:00:00Z
title: "What Worked (and What Didn't) When Training AEs and VAEs for Embedding Compression"
featured: false
draft: false
tags:
  - ML
  - Autoencoder
  - VAE
  - Embeddings
description: "Practical lessons from training autoencoders and VAEs for embedding compression—covering dimension choice, KL scheduling, contrastive signals, and evaluation metrics."
---

### TL;DR

- **Medium-dimensional latents** consistently offered the best quality–compression tradeoff. Going much lower hurt retrieval tasks; going much higher yielded diminishing returns.
- **VAE > plain AE for downstream retrieval.** With the right KL schedule, VAEs produced more robust latents than AEs trained only on reconstruction.
- **Stability hinges on KL pressure and learning rate.** Free-bits and a warmup schedule prevented KL collapse and NaNs while keeping useful structure.
- **A light contrastive signal helps.** A small latent-space alignment term (and optionally a queue) improved retrieval without sacrificing reconstruction.
- **Measure what matters.** Optimize for retrieval-style metrics, not just reconstruction loss.

---

### Dimension Choice

- Treat dimensionality as a first-class hyperparameter. In practice, a middle ground captured most of the essential structure with far smaller vectors.
- Extremely small latents dropped retrieval quality; very large latents added complexity with little observable benefit.

---

### AE vs VAE Objectives

- Plain AEs trained solely on reconstruction tended to memorize surface information and underperformed on retrieval-style evaluation.
- VAEs with moderate KL pressure yielded smoother, more semantically consistent latents that generalized better.
- The most reliable recipe combined: reconstruction loss + KL with warmup + free-bits to avoid KL collapse.

---

### Regularization and Stability

- Free-bits prevented the KL term from overwhelming individual latent dimensions and stabilized training.
- A short KL warmup allowed the decoder to learn useful signal before enforcing strong prior pressure.
- When instability appeared (e.g., NaNs), reducing learning rate and slightly increasing regularization resolved it consistently.

---

### Contrastive Signals (Kept Light)

- Adding a small contrastive component in latent space nudged representations toward retrieval-aware structure.
- A momentum/queue mechanism made the contrastive term more stable and sample-efficient, especially with larger batches.
- Overweighting the contrastive loss hurt reconstruction and did not yield better retrieval; conservative weights worked best.

---

### Optimization Practices That Stuck

- Prefer moderate learning rates and step-wise warmup over aggressive settings.
- Large effective batches improve stability and metric variance, but aren't strictly required if scheduling and regularization are solid.
- Gradient clipping and careful weight decay helped avoid rare spikes late in training.

---

### Evaluation Principles

- Prioritize retrieval-oriented metrics (pair similarity, query→item) over pure reconstruction.
- Always compare against a strong linear baseline (e.g., PCA) to sanity-check real gains from nonlinearity.
- Report a small, consistent metric suite across runs to keep decisions simple and comparable.

---

### A Practical Training Recipe

1. Start with a simple AE baseline to validate the pipeline and catch data issues quickly.
2. Switch to a VAE with: KL warmup, free-bits, and a moderate KL target.
3. If retrieval is the goal, add a small latent alignment term and, if available, a queue-based contrastive mechanism.
4. Pick a single promising dimensionality and tune learning rate and training length there before scaling out.
5. Lock the recipe and run a small set of confirmatory evaluations before expanding the search.

---

### Common Pitfalls (and Fixes)

| Pitfall | Fix |
|---------|-----|
| **KL collapse or degenerate latents** | Introduce free-bits and a gradual KL warmup |
| **Numerical instability (NaNs)** | Back off the learning rate, enable gradient clipping, verify normalization |
| **Overfitting to reconstruction** | Include retrieval-aware losses or lightweight contrastive terms; validate on retrieval tasks |
| **Chasing tiny gains** | Once a medium-dimensional setting performs well, extra dimensions rarely pay off |

---

### Closing Thought

For embedding compression, VAEs with carefully scheduled KL and light contrastive shaping repeatedly offered the strongest blend of compactness, stability, and downstream retrieval quality. Keep the objective balanced, the regularization principled, and the evaluation focused on the end task—not just reconstruction loss.

