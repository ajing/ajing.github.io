# Tool Selection Research Papers

A curated list of research papers on tool selection for LLM agents.

**Last Updated:** January 2026

> ⚠️ **Verification Note:** This document distinguishes between:
> - ✅ **Verified papers** (confirmed to exist via web search)
> - ⚠️ **Unverified arXiv IDs** (paper may exist but couldn't confirm)
> - 📝 **Synthesized content** (algorithm details/numbers I generated for illustration)

---

## Verified Papers

### 1. Toolformer: Language Models Can Teach Themselves to Use Tools ✅

| | |
|---|---|
| **arXiv** | [2302.04761](https://arxiv.org/abs/2302.04761) ✅ Verified |
| **Authors** | Timo Schick, et al. (Meta AI) |
| **Date** | February 2023 |

**Historical Significance:** THE foundational paper showing LLMs can learn tool use through self-supervision.

**Core Idea:** Let the model generate its own training data by:
1. Inserting candidate API calls into text
2. Executing calls and measuring if they improve perplexity
3. Keeping only calls that help
4. Fine-tuning on this self-generated data

**Tools Tested:** Calculator, Q&A, Search, Translation, Calendar

**Why It Matters:** Most subsequent tool learning papers build on this self-supervised approach.

---

### 2. AutoTool: Dynamic Tool Selection and Integration ✅

| | |
|---|---|
| **arXiv** | [2512.13278](https://arxiv.org/abs/2512.13278) ✅ Verified |
| **Authors** | Jiaru Zou, Ling Yang, et al. |
| **Date** | December 2025 |

**Core Idea:** Train LLMs with explicit tool-selection rationales using:
- 200,000 instances dataset with rationales
- Dual-phase optimization: SFT + RL
- Models trained: Qwen3-8B, Qwen2.5-VL-7B

**Key Insight:** Including rationales ("why this tool") in training data improves generalization to unseen tools.

---

### 3. Tool Learning with LLMs: A Survey ✅

| | |
|---|---|
| **arXiv** | [2304.08354](https://arxiv.org/abs/2304.08354) ✅ Verified |
| **Date** | 2024 |

**Taxonomy - Four Stages:**
1. Task Planning → decompose query
2. **Tool Selection** → choose tools (our focus)
3. Tool Calling → execute with parameters
4. Response Generation → synthesize outputs

---

### 4. ToolTalk: Evaluating Tool-Usage in Conversational Setting ✅

| | |
|---|---|
| **arXiv** | [2311.10775](https://arxiv.org/abs/2311.10775) ✅ Verified |

**Benchmark:** 28 tools, 7 plugins, multi-step tool usage in dialogue.

---

### 5. ConvDR: Few-Shot Conversational Dense Retrieval ✅

| | |
|---|---|
| **arXiv** | [2105.04166](https://arxiv.org/abs/2105.04166) ✅ Verified |

**Relevance:** Methods for conversation-aware retrieval that can apply to tool selection.

---

## Unverified Papers (arXiv IDs couldn't be confirmed)

> ⚠️ These papers were mentioned in my training or generated, but I could not verify the exact arXiv IDs via web search. The papers may exist under different IDs or with different titles.

### 6. "AutoTool-Graph" / Tool Usage Inertia ⚠️

| | |
|---|---|
| **arXiv** | 2511.14650 ⚠️ Unverified |

**Claimed Idea:** Graph-based tool selection using co-occurrence patterns from historical trajectories.

**Status:** Could not verify this specific arXiv ID. May be conflated with AutoTool-RL or a different paper.

---

### 7. "ToolScope" ⚠️

| | |
|---|---|
| **arXiv** | 2510.20036 ⚠️ Unverified |

**Claimed Idea:** Tool merging (clustering similar tools) + hybrid retrieval (BM25 + dense + cross-encoder).

**Status:** Could not verify this specific arXiv ID.

---

### 8. "TECTON" ⚠️

| | |
|---|---|
| **arXiv** | 2411.04535 ⚠️ Unverified |

**Claimed Idea:** Two-phase meta-reasoning (generate candidates, then reason about which is best).

**Status:** Could not verify this specific arXiv ID.

---

### 9. Bloomberg Joint Optimization ⚠️

| | |
|---|---|
| **Venue** | ACL 2025 Findings ⚠️ Unverified |

**Claimed Idea:** Co-optimize agent prompts and tool descriptions together.

**Status:** Could not verify the ACL anthology link.

---

### 10. ToolTweak / ToolHijacker ⚠️

| | |
|---|---|
| **arXiv** | 2510.02554, 2504.19793 ⚠️ Unverified |

**Claimed Idea:** Security attacks on tool selection (adversarial tool descriptions).

**Status:** Could not verify these specific arXiv IDs.

---

### 11. RapidTools ⚠️

| | |
|---|---|
| **Venue** | Unknown |

**Claimed Idea:** Empirical study on factors affecting tool selection (# tools, examples, model size).

**Status:** Could not verify source.

---

### 12. Meta-Cognition for Tool Use ⚠️

| | |
|---|---|
| **Venue** | Unknown |

**Claimed Idea:** Self-assessment to decide when NOT to use tools.

**Status:** Could not verify source.

---

## Industry Sources (Verified URLs)

### Manus AI ✅

| Blog | URL | Status |
|------|-----|--------|
| Context Engineering | https://manus.im/blog/context-engineering-for-ai-agents-lessons-from-building-manus | ✅ Real blog |
| Wide Research | https://manus.im/blog/wide-research-beyond-the-context-window | ✅ Real blog |

**Key Ideas from Manus:**
- Keep all tools in context, use logit masking instead of retrieval
- Tool naming conventions with prefixes (browser_*, file_*, shell_*)
- State machine to control which tools are valid per state
- Benefits: KV-cache stability, no retrieval errors

---

## Synthesized Content (For Illustration Only)

> 📝 The following are **patterns and implementations I synthesized** based on general knowledge. They are NOT from specific papers.

### Pattern: Hybrid Retrieval Pipeline

```
Query → BM25 (keyword) → Dense (semantic) → Cross-Encoder (rerank) → Top-K
```

This is a standard retrieval pattern, not specific to any paper.

### Pattern: Tool Description Template

```markdown
**Purpose:** One sentence
**Use when:** Conditions
**Do NOT use when:** Anti-conditions
**Input/Output:** Schema
```

This is a practical template, synthesized from general best practices.

### Pattern: Context-Aware Retrieval

Blending query with conversation history for "delete it" → "delete user John" resolution.

This is a general pattern from conversational search research, applied to tools.

---

## What's Real vs What I Made Up

| Item | Status |
|------|--------|
| Toolformer paper | ✅ Real |
| AutoTool-RL (2512.13278) | ✅ Real |
| Tool Learning Survey | ✅ Real |
| ToolTalk benchmark | ✅ Real |
| ConvDR | ✅ Real |
| Manus blogs | ✅ Real |
| AutoTool-Graph (2511.14650) | ⚠️ Unverified arXiv ID |
| ToolScope (2510.20036) | ⚠️ Unverified arXiv ID |
| TECTON (2411.04535) | ⚠️ Unverified arXiv ID |
| Bloomberg ACL link | ⚠️ Unverified |
| ToolTweak/ToolHijacker | ⚠️ Unverified arXiv IDs |
| RapidTools | ⚠️ Unverified |
| Meta-cognition paper | ⚠️ Unverified |
| Specific result numbers (30% reduction, etc.) | 📝 Synthesized |
| Algorithm pseudocode | 📝 Synthesized |
| Author names for unverified papers | 📝 Possibly incorrect |

---

## Research Gaps (Real)

These are actual gaps based on what verified research covers:

1. **Conversation-aware tool retrieval** - ConvDR exists for documents, not tools specifically
2. **Multi-provider tool routing** - No papers found
3. **Large API surface management** - No papers found
4. **Retrieval failure recovery** - No papers found
5. **Tool selection security defenses** - Attacks may exist, defenses unclear

---

## Recommendations

For the blog post, I recommend:

1. **Focus on verified papers:** Toolformer, AutoTool-RL, Tool Learning Survey, Manus blogs
2. **Mark synthesized patterns clearly:** Hybrid retrieval, description templates
3. **Remove or caveat unverified papers:** Don't cite arXiv IDs you can't verify
4. **Focus on ideas, not specific numbers:** The concepts are valid even if numbers aren't

---

## Citation Format (Verified Only)

```
Toolformer: [Schick et al., 2023](https://arxiv.org/abs/2302.04761)
AutoTool: [Zou et al., 2025](https://arxiv.org/abs/2512.13278)
Tool Learning Survey: [arXiv:2304.08354](https://arxiv.org/abs/2304.08354)
ToolTalk: [arXiv:2311.10775](https://arxiv.org/abs/2311.10775)
Manus: [manus.im/blog](https://manus.im/blog/context-engineering-for-ai-agents-lessons-from-building-manus)
```
