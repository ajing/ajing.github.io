---
author: Jing Lu
pubDatetime: 2026-01-10T00:00:00Z
title: "Tool Selection Optimization for LLM Agents at Scale"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Agents
description: "A deep technical dive into tool selection—retrieval strategies, context optimization, learned selection, and the engineering trade-offs that matter when scaling to hundreds of tools."
---

**The tool selection problem is deceptively simple:** given a user query and a set of available tools, pick the right ones. With 5 tools, this is trivial. With 500 tools, it becomes a critical bottleneck that determines whether your agent system works at all.

This post covers the technical approaches to tool selection optimization—from semantic retrieval to learned routing—with a focus on what actually works in production.

---

## 1. The Scale Problem

| # Tools | Selection Accuracy | Approach |
|---------|-------------------|----------|
| 5-10 | 90-95% | All in context |
| 20-50 | 80-90% | Good descriptions critical |
| 100-200 | 60-80% | Retrieval necessary |
| 500+ | 40-60% | Multi-stage selection |

**Why it matters:** Tool definitions cost 200-500 tokens each. At 500 tools, that's 200K+ tokens per request—**3-5 seconds** of inference time and **$45K/month** at scale. Optimized selection (10 tools) drops this to 400ms and $900/month.

**Bloomberg's finding:** Optimizing tool selection reduced unnecessary tool calls by **70%** while maintaining task success rates.

---

## 2. Retrieval-Based Tool Selection

### 2.1 Hybrid Retrieval

Pure semantic retrieval fails on vocabulary mismatch (user says "cancel" but tool says "refund"). Combine dense + sparse:

```python
class HybridToolRetriever:
    def __init__(self, tools, embedding_model, alpha=0.7):
        self.semantic = SemanticRetriever(tools, embedding_model)
        self.keyword = BM25Retriever(tools)
        self.alpha = alpha
    
    def retrieve(self, query: str, k: int = 10) -> List[Tool]:
        # Reciprocal Rank Fusion - no normalization needed
        tool_scores = defaultdict(float)
        for rank, tool in enumerate(self.semantic.retrieve(query, k*2)):
            tool_scores[tool.name] += 1 / (60 + rank)
        for rank, tool in enumerate(self.keyword.retrieve(query, k*2)):
            tool_scores[tool.name] += 1 / (60 + rank)
        
        sorted_tools = sorted(tool_scores.items(), key=lambda x: -x[1])
        return [self.get_tool(name) for name, _ in sorted_tools[:k]]
```

### 2.2 Context-Aware Retrieval (ToolScope)

ToolScope ([arXiv:2510.20036](https://arxiv.org/abs/2510.20036)) blends query embedding with conversation history:

```python
class ContextAwareRetriever:
    def retrieve(self, query: str, conversation_history: List[str], k: int = 10):
        query_emb = self.embedding_model.encode(query)
        
        if conversation_history:
            # Exponential decay for older turns
            history_embs = []
            for i, turn in enumerate(reversed(conversation_history[-5:])):
                decay = 0.8 ** i
                history_embs.append(self.embedding_model.encode(turn) * decay)
            
            context_emb = np.mean(history_embs, axis=0)
            combined_emb = 0.7 * query_emb + 0.3 * context_emb
        else:
            combined_emb = query_emb
        
        return self.retrieve_by_embedding(combined_emb, k)
```

A query like "now delete it" only makes sense with context from previous turns.

### 2.3 Embedding Model Selection

| Model | Cost/1M | Notes |
|-------|---------|-------|
| **Voyage 3 Large** | $0.18 | Top on Agentset tool retrieval benchmark |
| **text-embedding-3-large** | $0.13 | Balanced accuracy/cost |
| **BGE-M3** | $0.01 | Self-hosted, budget option |

**Note on Gemini Embedding:** Leads MTEB general benchmarks but underperforms on tool-specific retrieval (Agentset leaderboard). General embedding quality ≠ tool retrieval quality.

**Key insight from "Retrieval Models Aren't Tool-Savvy" (ACL 2025):** Standard embeddings achieve **<35% completeness@10** on tool retrieval. Fine-tuning with hard negatives (semantically similar but functionally different tools) significantly improves this.

---

## 3. Tool Description Optimization

### 3.1 Bloomberg's Context Optimization (ACL 2025)

**Key finding:** Jointly optimizing agent instructions AND tool descriptions reduced tool calls by **70%** (StableToolBench) and **47%** (RestBench) while maintaining pass rates.

**Benchmarks tested:**

| Benchmark | Description | Tools |
|-----------|-------------|-------|
| **StableToolBench** | Stability benchmark for tool-augmented LLMs | 16,000+ real APIs |
| **RestBench** | RESTful API evaluation | REST APIs |

**The insight:** Incomplete descriptions force LLMs to make exploratory calls. The optimized version adds:

```
When NOT to use: 
- If you already have the information from previous tool calls
- For structured data queries (use database_query instead)

Note: Each call costs ~200ms. Batch multiple intents into one call.
```

### 3.2 Description Template

```json
{
  "name": "weather_forecast",
  "description": "Get weather forecast for a location. Returns temperature, precipitation %, wind, conditions.",
  "when_to_use": "Weather, temperature, rain, outdoor planning queries",
  "when_not_to_use": "Historical data (use weather_history), air quality (use air_quality_api)",
  "parameters": {
    "location": {"type": "string", "description": "City name or coordinates"},
    "days": {"type": "integer", "default": 7, "description": "Forecast days (1-14)"}
  },
  "example_queries": ["What's the weather in Tokyo?", "Will it rain this weekend?"]
}
```

**Key elements:**
- `when_to_use` — triggers selection
- `when_not_to_use` — prevents over-selection  
- `example_queries` — improves retrieval matching

---

## 4. Tool Set Management

### 4.1 Tool Merging (ToolScope)

Large tool sets often have redundancy: `search_users` vs `find_users` vs `lookup_users`. Each redundant tool consumes tokens, creates ambiguity, and increases hallucination risk.

```python
class ToolMerger:
    def merge(self, tools: List[Tool], similarity_threshold=0.85) -> List[Tool]:
        embeddings = embed_tools(tools)
        clusters = self._cluster(embeddings, similarity_threshold)
        
        merged = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(tools[cluster[0]])
            else:
                # Merge into single tool with aliases
                primary = max([tools[i] for i in cluster], key=lambda t: len(t.description))
                aliases = [tools[i].name for i in cluster if tools[i].name != primary.name]
                primary.description += f"\n\nAliases: {', '.join(aliases)}"
                merged.append(primary)
        return merged
```

**Result:** 30-40% tool count reduction, improved selection accuracy.

### 4.2 Multi-Provider Abstraction

When multiple providers offer the same capability (weather: OpenWeatherMap/AccuWeather/WeatherAPI), expose a **single virtual tool** to the LLM:

```python
class VirtualToolRouter:
    def get_virtual_tools(self) -> List[Tool]:
        # LLM sees ONE tool per capability
        return [
            Tool(name="weather_forecast", 
                 description="Get weather forecast for any location"),
            Tool(name="web_search",
                 description="Search the web for current information"),
        ]
    
    def execute(self, tool_name: str, params: dict, strategy: str = "smart"):
        providers = self.providers[tool_name]
        
        if strategy == "cheapest":
            provider = min(providers, key=lambda p: p.cost)
        elif strategy == "reliable":
            provider = max(providers, key=lambda p: p.reliability)
        
        return self._call_with_fallback(provider, providers, params)
```

If user preference matters, add optional `provider` parameter with `default="auto"`.

**Key principle:** LLM thinks in **capabilities** ("I need weather data"), not implementations.

### 4.3 Large API Surfaces (Stripe, AWS, Salesforce)

100+ endpoints per service → can't fit in context. Solutions:

| Approach | Example | Token Overhead |
|----------|---------|----------------|
| **Hierarchical** | Domain → operation | ~25 tools max |
| **Intent-based** | "Charge customer" → `[get_customer, create_payment_intent, confirm_payment]` | ~10 intents |
| **CRUD abstraction** | `manage_customer(operation="create\|read\|update\|delete")` | ~15 resources |
| **Dynamic retrieval** | Embed API docs, retrieve k=5 per query | k retrieved |

---

## 5. Learned Tool Selection

### 5.1 AutoTool: Fine-Tuning for Selection (arXiv:2512.13278)

Trains **Qwen3-8B** and **Qwen2.5-VL-7B** using SFT + RL. Single model does both planning and tool selection.

**Phase 1:** Supervised learning on selection rationales (why tool X, not tool Y)

**Phase 2:** RL refinement with reward:
```python
reward = (
    task_success * 0.5 +
    tool_efficiency * 0.2 +      # Fewer tools = better
    no_hallucination * 0.2 +     # No invented tools
    correct_parameters * 0.1
)
```

### 5.2 Graph-Based Selection (arXiv:2511.14650)

Model tool co-occurrence as a graph:
- Nodes = Tools
- Edges = `P(tool_j | tool_i)` from historical trajectories

```python
class ToolSelectionGraph:
    def select_next(self, current_tools: List[str], query: str, k: int = 5):
        if not current_tools:
            # Use priors for first selection
            candidates = sorted(self.priors.items(), key=lambda x: -x[1])[:k]
        else:
            # Use transitions from last tool
            last_tool = current_tools[-1]
            candidates = sorted(self.adjacency[last_tool].items(), key=lambda x: -x[1])[:k]
        
        return self._rerank_by_query(candidates, query)
```

**Key insight:** Tool selection has **inertia**—certain combinations appear together. Learning these patterns reduces inference cost.

### 5.3 Constrained Decoding (Manus Approach)

Not retrieval, not training—**inference-time control**. Keep all tools in context, mask logits during generation.

| Technique | How It Works | Library |
|-----------|--------------|---------|
| **Logit masking** | Set disallowed tokens to `-inf` | Manus |
| **Grammar-constrained** | Force output to match CFG/regex | Outlines, LMQL, Guidance |
| **JSON schema** | Constrain to valid structure | OpenAI JSON mode, vLLM |

**Why not just remove tools?**

| Approach | Problem |
|----------|---------|
| Dynamically add/remove tools | Invalidates KV-cache |
| Retrieval-based filtering | Might filter out needed tools |
| **Logit masking** | Tools stay in context, output constrained |

**Implementation pattern (Manus):**
```python
# State machine controls which tool prefixes are allowed
allowed_prefixes = {
    "idle": ["browser_", "shell_", "search_"],
    "browsing": ["browser_"],  # Only browser tools while browsing
    "responding": [],          # No tools - must respond to user
}

def mask_logits(logits, state, tool_token_ids):
    allowed = [t for t in tool_token_ids if any(t.startswith(p) for p in allowed_prefixes[state])]
    for tool, token_id in tool_token_ids.items():
        if tool not in allowed:
            logits[:, token_id] = float('-inf')
    return logits
```

**When to use:** Tool availability depends on dynamic state. Avoids KV-cache invalidation from changing tool lists.

---

## 6. Multi-Stage Selection (1000+ tools)

**Category → Retrieve:** Fast classifier (~5ms) picks category, then retrieve within subset.

```
Query → Category Classifier → [Data|Web|Code|Doc] → Retrieve within category
                ↑
           <10ms, light model
```

**Multi-stage pipeline:**

| Stage | Method | Output | Latency |
|-------|--------|--------|---------|
| 1. Coarse | Embedding retrieval | k=100 | ~20ms |
| 2. Rerank | Cross-encoder | k=30 | ~50ms |
| 3. Filter | LLM confirmation | k=10 | ~100ms |

Trigger LLM stage only if top scores are ambiguous (gap < 0.1).

---

## 7. Error Recovery

**Fallback chain:** Learned selection → Retrieval → Category defaults → Universal tools (search, calculator, code_executor).

**Retry with exclusion:** On tool failure, exclude from next selection. Pre-map alternatives by capability:

```python
CAPABILITY_TOOLS = {
    "web_search": ["tavily", "serper", "google_search"],
    "weather": ["openweathermap", "weatherapi", "accuweather"],
    "code_execution": ["e2b_sandbox", "modal_sandbox", "local_docker"],
}
```

**Retrieval failure (vocabulary mismatch):** 
- Detect via low scores (top score < 0.5)
- Reformulate with synonyms: "cancel" → "refund", "terminate"
- Build alias index: user terms → tool terms

---

## 8. Production Recommendations

| Tool Count | Approach | Key Investment |
|------------|----------|----------------|
| **5-50** | All in context | Description quality |
| **50-200** | Semantic retrieval + reranking | Embedding fine-tuning |
| **200-1000** | Hierarchical routing | Category classifier, A/B testing |
| **1000+** | Learned selection + graph | Dedicated model, continuous learning |

---

## 9. Summary

**The 80/20 of tool selection:**

1. **Tool descriptions** (40%): Clear "when to use", "when not to use", examples
2. **Retrieval quality** (30%): Hybrid retrieval beats pure semantic
3. **Context awareness** (20%): Use conversation history
4. **Learned components** (10%): RL fine-tuning for edge cases

**Common mistakes:**
- Over-engineering retrieval when descriptions are bad
- Ignoring conversation context
- No fallback strategy
- Not monitoring selection accuracy in production

The best tool selection system is one where **you rarely think about it because it just works**.

---

## References

### Tool Selection Optimization
1. **ToolScope** - arXiv:2510.20036 - Tool merging and context-aware filtering
2. **AutoTool** - arXiv:2512.13278 - Dynamic tool selection via RL
3. **AutoTool (Graph)** - arXiv:2511.14650 - Historical trajectory modeling
4. **ToolBrain** - arXiv:2510.00023 - RL framework for tool use training
5. **Bloomberg ACL 2025** - Context optimization for tool calling (StableToolBench, RestBench)

### Tool Retrieval
6. **"Retrieval Models Aren't Tool-Savvy"** - ACL 2025 Findings - Shows <35% completeness@10
7. **HYRR: Hybrid Retrieval** - arXiv:2212.10528 - Combining BM25 with neural retrieval

### Constrained Decoding
8. **Manus** - [Context Engineering for AI Agents](https://medium.com/@peakji/context-engineering-for-ai-agents-lessons-from-building-manus-71883f0a67f2)
9. **Outlines** - github.com/outlines-dev/outlines - Grammar-constrained generation

*Code examples are synthesized implementations illustrating practical patterns.*
