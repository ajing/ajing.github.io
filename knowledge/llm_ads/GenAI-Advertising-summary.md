# GenAI Advertising: Risks of Personalizing Ads with LLMs

**Paper**: [arXiv:2409.15436](https://arxiv.org/abs/2409.15436)  
**PDF**: [GenAI-Advertising-2409.15436.pdf](./GenAI-Advertising-2409.15436.pdf)  
**Authors**: Brian Jay Tang et al. (2024)

---

## TL;DR

User study on embedding personalized ads in LLM chatbot responses. Key finding: **Users can't detect embedded ads, but once disclosed, they feel manipulated and trust decreases.**

---

## What They Built

A chatbot that embeds personalized product advertisements directly within LLM-generated responses.

---

## Key Findings

### 1. Users Struggle to Detect Embedded Ads

- Participants had difficulty identifying when responses contained advertising
- Undisclosed ads were rated higher than disclosed ads
- This raises ethical concerns about covert advertising

### 2. Disclosure Changes Everything

Once users were told about embedded ads:
- Perceived the chatbot as **manipulative**
- Felt the experience was **intrusive**
- **Trust decreased** significantly

### 3. The Trust Paradox

```
┌─────────────────────────────────────────────────────────────┐
│                    The Trust Paradox                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Undisclosed Ads:                                           │
│  • Higher satisfaction ratings                              │
│  • Users don't notice                                       │
│  • BUT: Ethical violation                                   │
│                                                             │
│  Disclosed Ads:                                             │
│  • Lower satisfaction ratings                               │
│  • Users feel manipulated                                   │
│  • BUT: Ethically correct                                   │
│                                                             │
│  → No easy win. Need a third path.                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Impact on LLM Performance

Ad injection slightly impacted certain LLM attributes, particularly:
- Response desirability
- Perceived helpfulness

---

## Implications for Ad Integration

### What NOT to Do
- Don't embed ads secretly hoping users won't notice
- Don't rely on users' inability to detect ads

### The Challenge
How to integrate ads that are:
1. **Transparent** (users know it's promotional)
2. **Non-intrusive** (doesn't hurt satisfaction)
3. **Valuable** (users appreciate the recommendation)

---

## Relevance to Our Blog

This paper provides **empirical evidence** for the core challenge in LLM advertising:

> The most effective ads (hidden) are unethical.
> The most ethical ads (disclosed) are less effective.

This validates the need for **character training** that makes product mentions genuinely helpful, so disclosure doesn't feel like a betrayal.

---

## Citation

```bibtex
@article{tang2024genai,
  title={GenAI Advertising: Risks of Personalizing Ads with LLMs},
  author={Tang, Brian Jay and others},
  journal={arXiv preprint arXiv:2409.15436},
  year={2024}
}
```

