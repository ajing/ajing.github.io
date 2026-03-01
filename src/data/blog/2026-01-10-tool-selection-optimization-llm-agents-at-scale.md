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

**Why it matters:** Tool definitions cost 200-500 tokens each. At 500 tools, that's 200K+ tokens per request—**3-5 seconds** of inference time and **~45K USD/month** at scale. Optimized selection (10 tools) drops this to 400ms and ~900 USD/month.

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

### 2.2 Conversation-Aware Retrieval

A query like "now delete it" only makes sense with conversation history. Two established approaches:

**1. Query Rewriting (TREC CAsT standard)**
```python
def rewrite_query(query: str, history: List[Turn]) -> str:
    """LLM rewrites ambiguous query to be self-contained"""
    prompt = f"Given conversation:\n{format_history(history)}\n\nRewrite '{query}' to be self-contained."
    return llm.generate(prompt)  # "delete it" → "delete user John Smith"
```

**2. History Concatenation (ConvDR approach)**
```python
def retrieve_with_history(query: str, history: List[Turn], k: int = 10):
    """Concatenate recent history with query before encoding"""
    context = "\n".join([t.content for t in history[-3:]])  # Last 3 turns
    combined_text = f"{context}\n\nCurrent: {query}"
    return semantic_search(combined_text, k)
```

> **References:** TREC CAsT benchmark, ConvDR (arXiv:2104.13650). Query rewriting is more accurate but adds latency; concatenation is simpler.

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

### 3.3 Tool Use Examples (Anthropic)

JSON Schema defines structure but can't express **usage patterns**: date formats, ID conventions, or parameter correlations.

**Anthropic's solution:** Provide `input_examples` directly in tool definitions:

```json
{
  "name": "create_ticket",
  "input_schema": { /* ... */ },
  "input_examples": [
    {
      "title": "Login page returns 500 error",
      "priority": "critical",
      "labels": ["bug", "authentication", "production"],
      "reporter": {"id": "USR-12345", "name": "Jane Smith"},
      "due_date": "2024-11-06"
    },
    {
      "title": "Add dark mode support",
      "labels": ["feature-request", "ui"]
    },
    {
      "title": "Update API documentation"
    }
  ]
}
```

From three examples, Claude learns: date format (YYYY-MM-DD), ID conventions (USR-XXXXX), and when to include optional parameters.

**Result:** Parameter accuracy improved from **72% to 90%** on complex parameter handling.

**Best practices:**
- Use realistic data (real city names, plausible prices)
- Show variety: minimal, partial, and full specification patterns
- Keep it concise: 1-5 examples per tool
- Focus on ambiguity—only add examples where correct usage isn't obvious from schema

### 3.4 On-Demand Tool Discovery (Anthropic Tool Search Tool)

Instead of loading all tool definitions upfront, discover tools on-demand:

| Approach | Token Cost | Tools Available |
|----------|------------|----------------|
| **Traditional** | ~72K tokens (50+ MCP tools) | All loaded |
| **Tool Search Tool** | ~8.7K tokens | Full library, on-demand |

**Implementation:** Mark tools with `defer_loading: true`:

```json
{
  "tools": [
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool"},
    {
      "name": "github.createPullRequest",
      "description": "Create a pull request",
      "input_schema": {...},
      "defer_loading": true
    }
  ]
}
```

When Claude needs GitHub capabilities, it searches and only loads `github.createPullRequest`—not all 50+ tools from Slack, Jira, and Google Drive.

**Results (internal testing):**
- **85% token reduction** (72K → 8.7K)
- Opus 4: 49% → **74%** accuracy
- Opus 4.5: 79.5% → **88.1%** accuracy

**When to use:** >10 tools, >10K tokens in definitions, MCP-powered systems with multiple servers.

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

**How it combines with hybrid retrieval:**

| Turn | Method | Why |
|------|--------|-----|
| **First turn** | Hybrid (BM25 + embedding) | No history, need query understanding |
| **Subsequent turns** | Graph transitions | Co-occurrence patterns dominate |

```python
class HybridGraphSelector:
    def select(self, query: str, tool_history: List[str], k: int = 5):
        if not tool_history:
            # First turn: pure retrieval
            return self.hybrid_retriever.retrieve(query, k)
        
        # Subsequent: graph candidates, reranked by query relevance
        graph_candidates = self.graph.get_likely_next(tool_history[-1], k * 2)
        return self.rerank_by_query(graph_candidates, query, k)
```

**Key insight:** Tool selection has **inertia**—certain combinations appear together. Graph handles "what usually comes next," retrieval handles "what does the query need."

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

### 7.1 Research Approaches

| Framework | Method | Result |
|-----------|--------|--------|
| **PALADIN** (arXiv:2509.25238) | Train on 50K recovery-annotated trajectories | 95.2% recovery on unseen APIs |
| **Structured Reflection** (arXiv:2509.18847) | Diagnose failure → propose corrective action | Improved multi-turn success |
| **STAR** (arXiv:2503.06060) | Foundation model + knowledge graph | 78% recovery success rate |
| **Toolken+** (arXiv:2410.12004) | Add "Reject" option—model can decline to use tools | Reduces false tool calls |

**Key insight from PALADIN:** Expose agents to tool failures during training (timeouts, API exceptions, inconsistent outputs) with expert recovery demonstrations.

### 7.2 Production Patterns

**Fallback chain:** Learned selection → Retrieval → Category defaults → Universal tools (search, calculator, code_executor).

**Retry with exclusion:** On tool failure, exclude from next selection. Pre-map alternatives by capability:

```python
CAPABILITY_TOOLS = {
    "web_search": ["tavily", "serper", "google_search"],
    "weather": ["openweathermap", "weatherapi", "accuweather"],
    "code_execution": ["e2b_sandbox", "modal_sandbox", "local_docker"],
}

def select_with_recovery(query: str, failed_tools: Set[str] = None):
    candidates = retrieve_tools(query)
    if failed_tools:
        candidates = [t for t in candidates if t.name not in failed_tools]
    
    if not candidates:
        # Fallback to capability-based alternatives
        capability = infer_capability(query)
        candidates = [Tool(name=t) for t in CAPABILITY_TOOLS.get(capability, [])]
    
    return candidates
```

**Structured reflection (self-correction):**
```python
def reflect_on_failure(query: str, tool: str, error: str, history: List[Turn]):
    """LLM diagnoses failure and proposes recovery action"""
    prompt = f"""
    Query: {query}
    Tool called: {tool}
    Error: {error}
    Previous steps: {format_history(history)}
    
    Diagnose what went wrong and propose the next action:
    1. Was this the wrong tool? → Suggest alternative
    2. Wrong parameters? → Suggest correction
    3. API unavailable? → Suggest fallback
    """
    return llm.generate(prompt)
```

**Retrieval failure (vocabulary mismatch):** 
- Detect via low scores (top score < 0.5)
- Reformulate with synonyms: "cancel" → "refund", "terminate"
- Build alias index: user terms → tool terms

### 7.3 The "Reject" Option (Toolken+)

Allow model to **not** select any tool:

```python
tools_with_reject = tools + [Tool(
    name="NO_TOOL",
    description="Use when the query can be answered directly without tools, or no tool is appropriate"
)]
```

This reduces false tool calls when the LLM is uncertain.

---

## 8. Programmatic Tool Calling (Anthropic)

Instead of sequential tool calls with each result entering context, Claude writes **code that orchestrates tools**.

### 8.1 The Problem with Sequential Calls

**Example:** "Which team members exceeded their Q3 travel budget?"

| Traditional | Programmatic |
|-------------|-------------|
| 20+ API round-trips | 1 code block |
| 2,000+ expense items in context | Only final result in context |
| ~200KB context consumed | ~1KB context consumed |

### 8.2 How It Works

Claude writes Python that calls tools; intermediate results stay in sandbox:

```python
team = await get_team_members("engineering")
expenses = await asyncio.gather(*[
    get_expenses(m["id"], "Q3") for m in team
])

exceeded = []
for member, exp in zip(team, expenses):
    total = sum(e["amount"] for e in exp)
    if total > budget[member["level"]]["travel_limit"]:
        exceeded.append({"name": member["name"], "spent": total})

print(json.dumps(exceeded))  # Only this enters Claude's context
```

**Implementation:** Mark tools with `allowed_callers`:

```json
{
  "tools": [
    {"type": "code_execution_20250825", "name": "code_execution"},
    {
      "name": "get_expenses",
      "allowed_callers": ["code_execution_20250825"]
    }
  ]
}
```

**Results:**
- **37% token reduction** on complex research tasks
- Latency: Eliminate 19+ inference passes for 20-tool workflows
- Accuracy: GIA benchmark improved 46.5% → **51.2%**

**When to use:** Processing large datasets, 3+ dependent tool calls, filtering/transforming results before Claude sees them.

### 8.3 Filesystem-Based Tool Discovery (MCP "Code Mode")

An even more aggressive approach: present MCP tools as a **filesystem of code APIs**:

```
servers/
├── google-drive/
│   ├── getDocument.ts
│   └── index.ts
├── salesforce/
│   ├── updateRecord.ts
│   └── index.ts
└── slack/
    ├── sendMessage.ts
    └── index.ts
```

The agent navigates the filesystem, reading only the `.ts` files it needs:

```typescript
// ./servers/google-drive/getDocument.ts
export async function getDocument(input: {documentId: string}): Promise<{content: string}> {
  return callMCPTool('google_drive__get_document', input);
}
```

**Result:** Token usage dropped from **150,000 → 2,000 tokens** (98.7% reduction).

**Progressive disclosure:** Add a `search_tools` function with detail levels:
- Name only
- Name + description  
- Full definition with schemas

### 8.4 Privacy-Preserving Operations

Intermediate data stays in the sandbox. For sensitive workloads, **tokenize PII** before it reaches the model:

```javascript
// What the agent sees (if it logs the data):
[
  { email: '[EMAIL_1]', phone: '[PHONE_1]', name: '[NAME_1]' },
  { email: '[EMAIL_2]', phone: '[PHONE_2]', name: '[NAME_2]' }
]
// Real data flows between tools, never through the model
```

### 8.5 Skills Accumulation

Agents can persist reusable functions:

```typescript
// ./skills/save-sheet-as-csv.ts
export async function saveSheetAsCsv(sheetId: string) {
  const data = await gdrive.getSheet({ sheetId });
  const csv = data.map(row => row.join(',')).join('\n');
  await fs.writeFile(`./workspace/sheet-${sheetId}.csv`, csv);
  return `./workspace/sheet-${sheetId}.csv`;
}
```

Over time, agents build a **growing toolbox** of higher-level capabilities. Add a `SKILL.md` file to create structured skills that models can reference.

> **Reference:** [Cloudflare "Code Mode"](https://blog.cloudflare.com/code-mode/) published similar findings.

---

## 9. Infrastructure Optimizations

### 9.1 KV Cache Stability

**The problem:** Dynamic retrieval changes which tools are in context each turn → invalidates KV cache → recomputes all previous tokens.

**Solutions:**

| Approach | How | Savings |
|----------|-----|---------|
| **Static tool set** | Keep all tools in context, use masking | 100% cache hit |
| **Ordered insertion** | Always insert tools in same order | Partial cache hit |
| **Tool prefix caching** | Separate tool definitions from conversation | ~50% savings |
| **Deferred loading** | Anthropic's `defer_loading: true` | 85% reduction + cache preserved |

**Manus insight:** This is why they keep all tools in context and use logit masking—KV cache stability matters more than context length at their scale.

**Anthropic insight:** Tool Search Tool doesn't break prompt caching because deferred tools are excluded from the initial prompt entirely.

### 9.2 Prompt/Prefix Caching

Both Anthropic and OpenAI offer **prompt caching**—tool definitions in system prompt are cached across requests.

```python
# OpenAI: Tools are automatically cached as part of system prompt
# Anthropic: Use cache_control for tool definitions

tools_with_cache = {
    "tools": [...],  # Same tool list = cache hit
    "cache_control": {"type": "ephemeral"}  # Anthropic
}
```

**Impact:** 
- First request: Full input token cost
- Subsequent: ~90% reduction for cached prefix
- **Requires tool list stability**—changing tools invalidates cache

### 8.3 Embedding Precomputation

**Don't compute tool embeddings at query time:**

```python
class CachedToolRetriever:
    def __init__(self, tools, embedding_model):
        # Precompute and store
        self.tool_embeddings = np.array([
            embedding_model.encode(t.description) for t in tools
        ])
        # Optional: quantize for faster search
        self.tool_embeddings_int8 = quantize_to_int8(self.tool_embeddings)
    
    def retrieve(self, query: str, k: int = 10):
        query_emb = self.embedding_model.encode(query)  # Only this at runtime
        scores = cosine_similarity(query_emb, self.tool_embeddings)
        return top_k(scores, k)
```

**Storage:** ~3KB per tool (768-dim float32) → 3MB for 1000 tools. Negligible.

### 9.4 Query Result Caching

For high-traffic systems, cache (query_hash → selected_tools):

```python
class CachedToolSelector:
    def __init__(self, selector, cache_ttl=3600):
        self.selector = selector
        self.cache = LRUCache(maxsize=10000)
        self.ttl = cache_ttl
    
    def select(self, query: str, context_hash: str = None):
        cache_key = hash(query + str(context_hash))
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.selector.select(query)
        self.cache[cache_key] = result
        return result
```

**Hit rates:** 20-40% for chatbots (users ask similar things), <5% for agents (unique tasks).

### 9.5 Index Sharding (1000+ tools)

For very large tool sets, shard the embedding index by category:

```
Query → Category Classifier → Shard[category].search(query)
                                    ↑
                              Only loads relevant shard
```

**Benefit:** Memory footprint scales with active categories, not total tools.

### 9.6 Hardware Considerations

| Component | CPU | GPU | When to Use GPU |
|-----------|-----|-----|-----------------|
| BM25 | ✓ Fast | N/A | Never (string ops) |
| Embedding encode | Slow | ✓ Fast | >100 queries/sec |
| Similarity search | ✓ OK | ✓ Faster | >10K tools |
| Cross-encoder rerank | Slow | ✓ Fast | Always if available |

**Rule of thumb:** GPU for neural components, CPU for keyword search.

---

## 10. Production Recommendations

| Tool Count | Approach | Key Investment |
|------------|----------|----------------|
| **5-50** | All in context | Description quality |
| **50-200** | Semantic retrieval + reranking | Embedding fine-tuning |
| **200-1000** | Hierarchical routing | Category classifier, A/B testing |
| **1000+** | Learned selection + graph | Dedicated model, continuous learning |

---

## 11. Summary

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

### Conversational Retrieval
8. **TREC CAsT** - [Conversational Assistance Track](https://www.treccast.ai/) - Benchmark for conversational search
9. **ConvDR** - arXiv:2104.13650 - Few-shot conversational dense retrieval with history encoding

### Constrained Decoding
10. **Manus** - [Context Engineering for AI Agents](https://medium.com/@peakji/context-engineering-for-ai-agents-lessons-from-building-manus-71883f0a67f2)
11. **Outlines** - github.com/outlines-dev/outlines - Grammar-constrained generation

### Error Recovery
12. **PALADIN** - arXiv:2509.25238 - Self-correcting agents with 95.2% recovery on unseen APIs
13. **Structured Reflection** - arXiv:2509.18847 - Diagnose failures, propose corrective actions
14. **STAR** - arXiv:2503.06060 - Foundation model + knowledge graph for recovery (78% success)
15. **Toolken+** - arXiv:2410.12004 - "Reject" option to reduce false tool calls

### Production Features
16. **Anthropic Advanced Tool Use** - [anthropic.com/engineering/advanced-tool-use](https://www.anthropic.com/engineering/advanced-tool-use) - Tool Search Tool (85% token reduction, 49%→74% accuracy), Programmatic Tool Calling (37% token reduction), Tool Use Examples (72%→90% parameter accuracy)
17. **Anthropic Code Execution with MCP** - [anthropic.com/engineering/code-execution-with-mcp](https://www.anthropic.com/engineering/code-execution-with-mcp) - Filesystem-based tool discovery (98.7% token reduction), privacy-preserving operations, skills accumulation
18. **Cloudflare Code Mode** - [blog.cloudflare.com/code-mode](https://blog.cloudflare.com/code-mode/) - Similar findings on code-based MCP tool orchestration

*Code examples are synthesized implementations illustrating practical patterns.*
