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

Perplexity introduced "sponsored related questions" that appear after the AI's answer:

```
User: "What's the best laptop for students?"

AI: [Detailed answer about laptop features...]

📌 Related questions:
• What are the best laptop deals under $500?  [Sponsored]
• How does the MacBook Air M3 compare?  [Sponsored by Apple]
• Which laptops have the best battery life?
```

**How it works**:
- Ads appear as suggested follow-up questions
- Clearly labeled as "Sponsored"
- Brands pay for placement in relevant query categories
- Users choose whether to click

**Why it works**: Low intrusiveness—the main answer is untouched, ads are optional next steps.

---

### 2. Sponsored Sources/Citations (Google AI Overviews)

**Status**: ✅ Live in Search

Google's AI Overviews include shopping ads and sponsored links in the source citations:

```
AI Overview: "The best running shoes for marathons include..."

Sources:
🛒 [Sponsored] Nike.com - Nike Vaporfly 3
🛒 [Sponsored] ASICS.com - Metaspeed Sky+
📄 Runner's World - 2024 Marathon Shoe Guide  
📄 Reddit r/running - Community recommendations
```

**How it works**:
- Existing Google Shopping ads surface in AI Overview citations
- Sponsored results mixed with organic sources
- Leverages existing ad auction infrastructure

---

### 3. Side Panel Ads (Microsoft Copilot/Bing)

**Status**: ✅ Live

Microsoft Copilot displays traditional Bing ads alongside AI responses:

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

**How it works**:
- Traditional display ads shown in sidebar
- Clear visual separation from AI content
- Leverages existing Bing Ads infrastructure

**Trade-off**: Lower engagement but maximum transparency.

---

### 4. Affiliate Links in Responses (Various)

**Status**: ✅ Live in multiple products

Many AI assistants include affiliate links when recommending products:

```
User: "What's a good beginner guitar?"

AI: "For beginners, I recommend the Yamaha FG800. 
You can find it on [Amazon](https://amazon.com/dp/...) for around $200."
```

**How it works**:
- AI includes product links with affiliate tracking
- Revenue share when users purchase
- Often not explicitly disclosed as advertising

**Examples**: Various AI shopping assistants, review sites with AI chat.

---

### 5. Brand Partnerships/Sponsored Content (ChatGPT Plugins Era)

**Status**: ⚠️ Limited/Deprecated

During the ChatGPT plugins era, brands like Expedia, Instacart, and Kayak had direct integrations:

```
User: "Find me a flight to Tokyo"

AI: [Powered by Kayak]
"I found several options for you:
• ANA - $850 round trip, 1 stop
• JAL - $920 round trip, direct
[Book on Kayak →]"
```

**Current status**: Plugins deprecated in favor of GPTs and Actions, but brand partnerships continue in various forms.

---

## In Research / Experimental

### 1. Contextual In-Response Ad Injection

**Status**: 🔬 Research (GEM-Bench)

Embedding ads directly within the AI's response text:

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

**Why not in production yet**: User trust concerns—studies show users feel manipulated when ads are embedded without clear disclosure.

---

### 2. Commercial Intent Retrieval (RARE Framework)

**Status**: 🔬 Research → Production at scale (Alibaba)

The [RARE paper](https://arxiv.org/pdf/2504.01304.pdf) introduced "Commercial Intentions" for real-time ad retrieval:

```
User Query: "comfortable shoes for standing all day"
     ↓
LLM generates Commercial Intent: "ergonomic work footwear, 
cushioned insoles, nurse shoes, retail worker shoes"
     ↓
Retrieved Ads: [Nike Air Max, Crocs Work, Skechers Go Walk]
```

**Production results** (deployed in search advertising):
- +5.04% consumption
- +6.37% GMV
- +1.28% CTR

**Status note**: While the paper is research, the system is deployed in Alibaba's ad infrastructure.

---

### 3. Generative Engine Optimization (Rewrite-to-Rank)

**Status**: 🔬 Research (arXiv:2507.21099)

The inverse of ad injection—optimizing brand content to be **cited by LLMs**:

```
Traditional SEO:           GEO (Generative Engine Optimization):
────────────────           ────────────────────────────────────
Optimize for Google        Optimize for LLM retrieval
Keyword density            Semantic richness
Meta tags                  Answer implicit questions
Backlinks                  Context that triggers citations
```

**Example transformation**:

| Before | After (GEO-optimized) |
|--------|----------------------|
| "Nike Air Max 90 - Air cushioning" | "For users seeking comfortable everyday running shoes with responsive cushioning, the Nike Air Max 90 offers visible Air technology ideal for extended wear" |

**Key insight**: The second version is more likely to be retrieved AND cited by RAG-based LLMs because it directly answers potential user queries.

**Why it matters**: Brands are already investing in GEO to ensure their products appear in AI-generated responses—this is the supply side of the GEM ecosystem.

---

### 4. LLM-Generated Persuasive Ads

**Status**: 🔬 Research (arXiv:2512.03373)

Using LLMs not just to inject ads, but to **generate more persuasive ad content**:

**Key findings**:
- LLM-generated ads match human ads in **personalization** (Big Five personality targeting)
- LLM-generated ads **outperform humans** in persuasive storytelling
- Particularly effective: authority appeals, social proof, aspirational narratives

**Implication**: When LLMs do recommend products, they can make those recommendations significantly more compelling than human-written copy.

---

### 5. The Trust Paradox (GenAI Advertising Research)

**Status**: 🔬 Research (arXiv:2409.15436)

User study findings that explain why in-response ads aren't widely deployed:

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

**Why this matters**: This explains the current production landscape—companies opt for clearly separated ads (Perplexity's follow-up questions, Google's sponsored sources) rather than embedded ads because the trust cost is too high.

---

## Summary: Production vs. Research

| Format | Status | Company/Paper |
|--------|--------|---------------|
| Sponsored follow-up questions | ✅ Production | Perplexity |
| Sponsored sources/citations | ✅ Production | Google AI Overviews |
| Side panel ads | ✅ Production | Microsoft Copilot |
| Affiliate links | ✅ Production | Various |
| Brand plugin partnerships | ⚠️ Limited | OpenAI (deprecated plugins) |
| In-response ad injection | 🔬 Research | GEM-Bench (arXiv:2509.14221) |
| Commercial intent retrieval | 🔬→✅ Research→Prod | RARE (arXiv:2504.01304) |
| Generative Engine Optimization | 🔬 Research | Rewrite-to-Rank (arXiv:2507.21099) |
| LLM-generated persuasive ads | 🔬 Research | arXiv:2512.03373 |
| Trust/disclosure dynamics | 🔬 Research | GenAI Advertising (arXiv:2409.15436) |

---

## The Trend

Production systems favor **separation**:
- Ads clearly labeled and visually distinct
- User choice to engage or ignore
- Leveraging existing ad infrastructure

Research explores **integration**:
- Ads woven into response text
- Personalized and contextual
- New retrieval and generation mechanisms

The gap between research and production reflects the **trust problem**: users accept clearly-labeled ads but feel manipulated by hidden ones. The winning format will likely be one that's both integrated AND transparent.

---

## References

### Core Papers

- [GEM-Bench: A Benchmark for Ad-Injected Response Generation](https://arxiv.org/pdf/2509.14221.pdf) (arXiv:2509.14221) - First comprehensive benchmark for evaluating ad injection in LLM responses
- [RARE: Real-time Ad Retrieval via LLM-generative Commercial Intention](https://arxiv.org/pdf/2504.01304.pdf) (arXiv:2504.01304) - Production-validated approach for commercial intent-based ad retrieval
- [GenAI Advertising: Risks of Personalizing Ads with LLMs](https://arxiv.org/pdf/2409.15436.pdf) (arXiv:2409.15436) - User study on trust and manipulation perception

### Additional Research

- [Rewrite-to-Rank: Optimizing Ad Visibility in LLM Retrieval](https://arxiv.org/abs/2507.21099) (arXiv:2507.21099) - Generative Engine Optimization techniques
- [LLM-Generated Ads: From Personalization Parity to Persuasion Superiority](https://arxiv.org/abs/2512.03373) (arXiv:2512.03373) - LLM capabilities in ad content generation
