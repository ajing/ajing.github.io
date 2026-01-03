# GEM-Bench: A Benchmark for Ad-Injected Response Generation

**Paper**: [arXiv:2509.14221](https://arxiv.org/abs/2509.14221)  
**PDF**: [GEM-Bench-2509.14221.pdf](./GEM-Bench-2509.14221.pdf)  
**Authors**: Silan Hu, Shiqi Zhang, Yimin Shi, Xiaokui Xiao (2025)

---

## TL;DR

First comprehensive benchmark for evaluating **ad-injected LLM responses** in "Generative Engine Marketing" (GEM). Key finding: simple prompt-based ads get clicks but hurt satisfaction; post-generation injection is better but slower.

---

## What is GEM (Generative Engine Marketing)?

An emerging ecosystem for **monetizing LLM-based chatbots** by seamlessly integrating relevant advertisements into their responses. Think of it as the evolution of Search Engine Marketing (SEM) for the AI era.

---

## The Problem

Existing benchmarks are **not designed** for evaluating ad-injected responses, which limits research progress in this area.

---

## GEM-Bench Components

### 1. Three Curated Datasets

| Dataset Type | Coverage |
|--------------|----------|
| Real-world product catalogs | Actual brand/product data |
| Human-curated prompts | Natural user queries |
| Large-scale market scenarios | Diverse ad contexts |

Covers both **chatbot** and **search** scenarios.

### 2. Metric Ontology (6 Dimensions)

| Dimension | What it measures |
|-----------|------------------|
| **Trust** | Does the response feel trustworthy? |
| **Naturalness** | Is the ad integration seamless? |
| **Personality** | Does the character feel authentic? |
| **Relevance** | Is the ad contextually appropriate? |
| **Coherence** | Does the response flow logically? |
| **Fluency** | Is the language natural? |

### 3. Baseline Solutions

| Model | Approach | Description |
|-------|----------|-------------|
| **AdLLM** | RAG-based | Retrieval-augmented generation for ad injection |
| **AdChat** | Conversational | End-to-end conversational ad model |

Built within an **extensible multi-agent framework**.

---

## Key Findings

### The Core Trade-off

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Simple Prompt-Based Injection                             │
│   ├── ✅ Achieves reasonable engagement (CTR)               │
│   └── ❌ Reduces user satisfaction                          │
│                                                             │
│   Post-Generation Injection (insert into ad-free response)  │
│   ├── ✅ Better user satisfaction                           │
│   └── ❌ Additional computational overhead                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implications

1. **No free lunch**: You can't maximize both engagement AND satisfaction with naive approaches
2. **Need for innovation**: More effective and efficient solutions are needed
3. **User experience matters**: Satisfaction is as important as click-through rates

---

## Relevance to Our Work

This paper validates several key points:

1. **GEM is a real, academically recognized field** - not just marketing buzzword
2. **The satisfaction vs. engagement trade-off is real** - need thoughtful design
3. **RAG-based approaches (AdLLM) are viable** - retrieval for fresh product info works
4. **Multi-dimensional evaluation is necessary** - trust, naturalness, etc. all matter

---

## Open Questions for Future Research

- How to achieve high engagement WITHOUT sacrificing satisfaction?
- Can character training improve naturalness scores?
- What's the right disclosure strategy for maintaining trust?
- How to minimize computational overhead of post-generation injection?

---

## Citation

```bibtex
@article{hu2025gembench,
  title={GEM-Bench: A Benchmark for Ad-Injected Response Generation within Generative Engine Marketing},
  author={Hu, Silan and Zhang, Shiqi and Shi, Yimin and Xiao, Xiaokui},
  journal={arXiv preprint arXiv:2509.14221},
  year={2025}
}
```

