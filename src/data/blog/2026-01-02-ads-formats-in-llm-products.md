---
author: Jing Lu
pubDatetime: 2026-01-02T00:00:00Z
title: "Ad Formats in LLM Products: What's Live vs. What's Research"
featured: false
draft: false
tags:
  - AI
  - LLM
  - Advertising
  - Product
description: "A survey of advertising formats in LLM products—separating what's deployed in production from what remains in research."
---

As LLM products mature, monetization through advertising is becoming inevitable. This post surveys the landscape of ad formats—separating what's already deployed in production from what remains in the research stage.

---

## In Production

### 1. Sponsored Follow-Up Questions (Perplexity)

**Status**: ✅ Live since November 2024

```
User: "What's the best laptop for students?"

AI: [Detailed answer about laptop features...]

📌 Related questions:
• What are the best laptop deals under $500?  [Sponsored]
• How does the MacBook Air M3 compare?  [Sponsored by Apple]
• Which laptops have the best battery life?
```

**Why it works**: Low intrusiveness—the main answer is untouched, ads are optional next steps.

---

### 2. Sponsored Sources/Citations (Google AI Overviews)

**Status**: ✅ Live in Search

```
AI Overview: "The best running shoes for marathons include..."

Sources:
🛒 [Sponsored] Nike.com - Nike Vaporfly 3
🛒 [Sponsored] ASICS.com - Metaspeed Sky+
📄 Runner's World - 2024 Marathon Shoe Guide  
```

---

### 3. Side Panel Ads (Microsoft Copilot/Bing)

**Status**: ✅ Live

```
┌─────────────────────────────────────────┬──────────────────┐
│                                         │                  │
│  Copilot response about coffee makers   │  [Ad]            │
│  comparing features and prices...       │  Nespresso       │
│                                         │  VertuoPlus      │
│                                         │  $199 → $149     │
│                                         │  Shop Now →      │
│                                         │                  │
└─────────────────────────────────────────┴──────────────────┘
```

**Trade-off**: Lower engagement but maximum transparency.

---

## In Research / Experimental

### 1. Contextual In-Response Ad Injection

**Status**: 🔬 Research (GEM-Bench)

```
User: "Plan a 3-day trip to Paris"

AI: "Day 1: Start at the Eiffel Tower. For a unique experience, 
[GetYourGuide](https://getyourguide.com) offers skip-the-line 
tickets with local guides... Day 2: Visit the Louvre..."
```

**Research findings** (GEM-Bench, arXiv:2509.14221):
- Simple prompt injection: Good CTR, but reduced user satisfaction
- Post-generation refinement: Better UX, but adds computational overhead
- Key challenge: Making ads feel natural, not intrusive

---

### 2. The Trust Paradox

**Status**: 🔬 Research (arXiv:2409.15436)

```
┌─────────────────────────────────────────────────────────────┐
│                    The Trust Paradox                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FINDING 1: Users can't detect embedded ads                 │
│  FINDING 2: Undisclosed ads get higher satisfaction ratings │
│  FINDING 3: Once disclosed, users feel manipulated          │
│  FINDING 4: Trust decreases significantly after disclosure  │
│                                                             │
│  → No easy win. Hidden ads = unethical.                     │
│  → Disclosed ads = lower satisfaction.                      │
│  → Need a third path: genuinely helpful recommendations.    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

| Format | Status | Company/Paper |
|--------|--------|---------------|
| Sponsored follow-up questions | ✅ Production | Perplexity |
| Sponsored sources/citations | ✅ Production | Google AI Overviews |
| Side panel ads | ✅ Production | Microsoft Copilot |
| In-response ad injection | 🔬 Research | GEM-Bench |
| Trust/disclosure dynamics | 🔬 Research | GenAI Advertising |

**The Trend**: Production systems favor **separation** (clearly labeled ads), while research explores **integration** (ads woven into responses). The gap reflects the **trust problem**: users accept clearly-labeled ads but feel manipulated by hidden ones.

---

## References

- [GEM-Bench](https://arxiv.org/pdf/2509.14221.pdf) (arXiv:2509.14221) - Benchmark for ad injection evaluation
- [GenAI Advertising](https://arxiv.org/pdf/2409.15436.pdf) (arXiv:2409.15436) - User study on trust and manipulation
- [RARE](https://arxiv.org/pdf/2504.01304.pdf) (arXiv:2504.01304) - Commercial intent-based ad retrieval

