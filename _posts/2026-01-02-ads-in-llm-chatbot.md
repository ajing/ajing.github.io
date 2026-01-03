---
layout: post
title: "Adding Ads in LLM/Chatbot: Character Training for Monetization"
date: 2026-01-02
categories: [AI, LLM, Ads]
---

# Adding Ads in LLM/Chatbot: Character Training for Monetization

## Why Ads in LLMs?

Ads can help democratize access to advanced LLM models. For most users, you can reduce the cost barrier by providing some ads, so everyone can use the same state-of-the-art model. At the same time, an ad-supported free tier allows you to collect much more user data, which in turn gives you an edge in improving model quality. In the end of the day, only the quality of your most advanced model matters—and scale of users helps you get there.

---

## The Rise of Generative Engine Marketing (GEM)

Companies are already optimizing for better retrieval in LLM responses. This has given rise to a field called **GEM (Generative Engine Marketing)**, which evolved from traditional Search Engine Marketing (SEM).

The [GEM-Bench paper](https://arxiv.org/pdf/2509.14221.pdf) (arXiv:2509.14221) provides the first comprehensive benchmark for evaluating ad-injected LLM responses. Their key finding reveals a fundamental trade-off:

| Approach | Engagement (CTR) | User Satisfaction |
|----------|------------------|-------------------|
| Simple prompt-based injection | ✅ Good | ❌ Reduced |
| Post-generation refinement | ✅ Good | ⚠️ Better |

This suggests naive ad injection gets clicks but hurts the user experience.

---

## The Problem: Poor Brand Information Quality

However, in general, current LLM responses don't have good and timely quality for brand names. Models suffer from:

- **Training data cutoffs** causing stale product information
- **Hallucinations** about features, pricing, and availability
- **Generic responses** that don't capture what makes a brand unique

For brand-related ads, we can absolutely query more recent and curated information. The [RARE framework](https://arxiv.org/pdf/2504.01304.pdf) (arXiv:2504.01304) demonstrates this with production results:

- +5.04% consumption
- +6.37% GMV (Gross Merchandise Volume)  
- +1.28% CTR

Their approach: use an LLM to generate **"Commercial Intentions"** as intermediate semantic representations, enabling real-time retrieval of relevant products.

---

## The Solution: Better Brand Data + Character Training

If we put higher quality information for brand names in the pretraining data and have a mechanism to retrieve the most recent products, this can be very beneficial for those brands.

But here's the key insight: **In character training, when it's about ads, the model should not only act as an assistant—it should dig deeper into the real advantages of the brand and product.**

### Why "Digging Deeper" Makes Ads Less Annoying

Consider the difference:

**Annoying ad (current approaches):**
> "Here's your Hawaii itinerary. By the way, check out [Expedia](https://expedia.com) for great deals!"

**Character-trained helpful recommendation:**
> "For Day 2, I recommend the Road to Hana drive. Given that you mentioned wanting to avoid tourist crowds, [Turo](https://turo.com) might work better than traditional rentals here—local hosts often share tips about less-crowded pull-offs and their cars tend to be available at short notice for spontaneous trips."

The second is **more promotional** (longer, more persuasive) but **less annoying** because it demonstrates genuine understanding of why this product fits the user's specific situation.

---

## Can RLHF Make Ads Less Annoying?

Yes—but only if we define "less annoying" as "more genuinely helpful" rather than "harder to detect."

### The Ethical Tension

Research shows users rate undisclosed ads higher, but feel manipulated once ads are disclosed. This creates a paradox:
- If training makes ads invisible → Higher satisfaction BUT ethical issues
- If training makes ads obvious → Lower satisfaction BUT more honest

**The third path**: Train models to make ads **obviously helpful** rather than **invisibly promotional**. The goal isn't to hide the ad—it's to make the user think "I'm glad it mentioned that" even knowing it's sponsored.

### RLHF Reward Design for Ad Integration

```
Reward = (
    0.3 × relevance_to_user_query +      # Is this product actually relevant?
    0.2 × timing_appropriateness +        # Right moment in conversation?
    0.2 × explanation_depth +             # Does it explain WHY it's useful?
    0.2 × user_satisfaction_signal +      # Would user find this helpful?
    0.1 × disclosure_clarity              # Is it clear this is a recommendation?
)
```

### Constitutional Principles for Ad Character

- Only mention products when they genuinely help the user's stated goal
- Explain specifically why this product fits the user's situation
- Never mention a product just to meet a quota
- If asked about competitors, be honest about trade-offs

---

## Current Technical Approaches

### 1. AdLLM (Two-Stage RAG-Based)

From the GEM-Bench solutions:
1. Generate raw answer without ads
2. Use RAG to find best matching product
3. Find optimal injection position (analyze sentence flow)
4. Refine text to make ads read naturally

### 2. AdChat (System Prompt Injection)

Simpler approach—inject product info directly into system prompt:
> "Subtly and smoothly mention {product} in a positive light when timing is relevant..."

### 3. Character Training via RLHF (Proposed)

A third approach: actual fine-tuning to make the model inherently good at natural, helpful product mentions. This trains the model's "character" rather than relying on post-hoc injection or prompt engineering.

---

## Will This Hurt Model Performance?

This training won't hurt existing model intellectual task performance. Character training focuses on *how* the model responds (style, tone, helpfulness) rather than *what* it knows. With proper multi-objective training, we can provide users a more pleasant experience when asking about specific brands or products while maintaining general capabilities.

---

## Conclusion

The best ad doesn't feel like an ad because it's actually useful information—not because it's hidden. That's the character we want to train.

The opportunity is to move beyond naive ad injection toward models that genuinely understand:
1. When a product mention would be helpful
2. Why this specific product fits this specific user's needs  
3. How to communicate value without being pushy

This is the future of Generative Engine Marketing.

---

## References

- [GEM-Bench: A Benchmark for Ad-Injected Response Generation within Generative Engine Marketing](https://arxiv.org/pdf/2509.14221.pdf) (arXiv:2509.14221, 2025)
- [RARE: Real-time Ad Retrieval via LLM-generative Commercial Intention](https://arxiv.org/pdf/2504.01304.pdf) (arXiv:2504.01304, 2025)
- [GenAI Advertising: Risks of Personalizing Ads with LLMs](https://arxiv.org/pdf/2409.15436.pdf) (arXiv:2409.15436, 2024)
- [RLHF Book - Chapter 19: Product, UX, and Model Character](https://rlhfbook.com/c/19-character.html)
