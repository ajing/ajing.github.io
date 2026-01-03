---
author: Jing Lu
pubDatetime: 2025-01-02T00:00:00Z
title: "Adding Ads in LLM/Chatbot: Character Training for Monetization"
featured: false
draft: false
tags:
  - AI
  - LLM
  - Advertising
  - RLHF
description: "Exploring how to integrate ads in LLMs through character training—making recommendations genuinely helpful rather than annoyingly promotional."
---

## Why Ads in LLMs?

Ads can help democratize access to advanced LLM models. For most users, you can reduce the cost barrier by providing some ads, so everyone can use the same state-of-the-art model. At the same time, an ad-supported free tier allows you to collect much more user data, which gives you an edge in improving model quality.

---

## The Rise of Generative Engine Marketing (GEM)

Companies are already optimizing for better retrieval in LLM responses. This has given rise to **GEM (Generative Engine Marketing)**, which evolved from traditional Search Engine Marketing (SEM).

The [GEM-Bench paper](https://arxiv.org/pdf/2509.14221.pdf) provides the first comprehensive benchmark for evaluating ad-injected LLM responses:

| Approach | Engagement (CTR) | User Satisfaction |
|----------|------------------|-------------------|
| Simple prompt-based injection | ✅ Good | ❌ Reduced |
| Post-generation refinement | ✅ Good | ⚠️ Better |

This suggests naive ad injection gets clicks but hurts the user experience.

---

## The Problem: Poor Brand Information Quality

Current LLM responses don't have good and timely quality for brand names. Models suffer from:

- **Training data cutoffs** causing stale product information
- **Hallucinations** about features, pricing, and availability
- **Generic responses** that don't capture what makes a brand unique

The [RARE framework](https://arxiv.org/pdf/2504.01304.pdf) demonstrates better approaches with production results:
- +5.04% consumption
- +6.37% GMV
- +1.28% CTR

---

## The Solution: Better Brand Data + Character Training

If we put higher quality information for brand names in the pretraining data and have a mechanism to retrieve recent products, this can be beneficial for brands.

**Key insight**: In character training, when it's about ads, the model should not only act as an assistant—it should dig deeper into the real advantages of the brand and product.

### Why "Digging Deeper" Makes Ads Less Annoying

**Annoying ad:**
> "Here's your Hawaii itinerary. By the way, check out [Expedia](https://expedia.com) for great deals!"

**Character-trained helpful recommendation:**
> "For Day 2, I recommend the Road to Hana drive. Given that you mentioned wanting to avoid tourist crowds, [Turo](https://turo.com) might work better than traditional rentals here—local hosts often share tips about less-crowded pull-offs."

The second is **more promotional** but **less annoying** because it demonstrates genuine understanding of why this product fits the user's situation.

---

## Can RLHF Make Ads Less Annoying?

Yes—but only if we define "less annoying" as "more genuinely helpful."

### The Ethical Tension

Research shows users rate undisclosed ads higher, but feel manipulated once disclosed. This creates a paradox:
- Hidden ads → Higher satisfaction BUT ethical issues
- Obvious ads → Lower satisfaction BUT more honest

**The third path**: Train models to make ads **obviously helpful** rather than **invisibly promotional**.

### RLHF Reward Design

```python
reward = (
    0.3 * relevance_to_user_query +
    0.2 * timing_appropriateness +
    0.2 * explanation_depth +
    0.2 * user_satisfaction_signal +
    0.1 * disclosure_clarity
)
```

### Constitutional Principles

- Only mention products when they genuinely help the user's stated goal
- Explain specifically why this product fits the user's situation
- Never mention a product just to meet a quota
- If asked about competitors, be honest about trade-offs

---

## Current Technical Approaches

### 1. AdLLM (Two-Stage RAG-Based)

1. Generate raw answer without ads
2. Use RAG to find best matching product
3. Find optimal injection position
4. Refine text to make ads read naturally

### 2. AdChat (System Prompt Injection)

Inject product info directly into system prompt:
> "Subtly mention {product} in a positive light when timing is relevant..."

### 3. Character Training via RLHF (Proposed)

Actual fine-tuning to make the model inherently good at natural, helpful product mentions.

---

## Will This Hurt Model Performance?

No—character training focuses on *how* the model responds (style, helpfulness) rather than *what* it knows. With proper multi-objective training, we can provide users a more pleasant experience when asking about brands while maintaining general capabilities.

---

## Conclusion

The best ad doesn't feel like an ad because it's actually useful information—not because it's hidden. That's the character we want to train.

---

## References

- [GEM-Bench](https://arxiv.org/pdf/2509.14221.pdf) (arXiv:2509.14221)
- [RARE](https://arxiv.org/pdf/2504.01304.pdf) (arXiv:2504.01304)
- [GenAI Advertising](https://arxiv.org/pdf/2409.15436.pdf) (arXiv:2409.15436)
- [RLHF Book - Chapter 19: Product, UX, and Model Character](https://rlhfbook.com/c/19-character.html)

