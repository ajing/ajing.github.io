---
author: Jing Lu
pubDatetime: 2026-03-01T00:00:00Z
title: "Trajectory-Augmented In-Context Learning: A Training-Free Alternative to RL Post-Training"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Agents
  - RAG
description: "What if you could get RL post-training-level performance without updating a single parameter? Trajectory-augmented ICL retrieves high-quality reasoning traces at inference time, offering a faster, cheaper, and continuously improving alternative to GRPO/PPO."
---

**RL post-training works.** GRPO and PPO can teach models to reason, use tools, and recover from errors. But it's expensive — significant GPU hours, careful reward engineering, and every policy update requires re-sampling trajectories from scratch.

What if there's a simpler path? **Store successful reasoning trajectories in a database, retrieve the most relevant ones at inference time, and guide the model through in-context learning — no parameter updates required.**

This post explores *Trajectory-Augmented ICL*, a training-free approach that can match or complement RL post-training while offering continuous improvement through a self-reinforcing flywheel.

---

## The Core Idea

The standard RL post-training loop collects verifiable trajectories (thought → action → observation → reward) and updates model parameters via GRPO/PPO. It's effective but costly — and each policy iteration is a one-shot affair.

Our alternative flips this: keep the model frozen, but make it *smarter at test time* by showing it how similar problems were solved before.

```
User Query → Trajectory Retriever → Trajectory Database → Top-K Similar Trajectories
   → Context Assembly → LLM Inference → Task Completion
```

The key comparison:

| Method | Parameter Update | Compute Cost | Continuous Learning |
|--------|-----------------|--------------|---------------------|
| **RL Post-Training** (GRPO/PPO) | ✅ Full / LoRA | 🔴 High (GPU hours) | ❌ Retrain needed |
| **Test-Time Training** (TTT) | ✅ Temporary | 🟡 Medium | ❌ Per-query only |
| **Trajectory ICL** (ours) | ❌ Frozen params | 🟢 Low (retrieval + inference) | ✅ Grows with usage |

---

## Prior Art: Who's Already Doing This?

This idea doesn't come from nowhere. Several research threads converge here:

**Agent Experiential Learning:**

| Work | Core Idea |
|------|-----------|
| **ExpeL** (Zhao et al., 2023) | Agent collects success/failure experiences; retrieves past experiences at inference via ICL |
| **Voyager** (Wang et al., 2023) | Minecraft agent maintains a skill library of reusable skills, retrieved by similarity |
| **Reflexion** (Shinn et al., 2023) | Agent extracts verbal reflections from failures, stored in episodic memory |
| **CER** (2024) | Training-free dynamic memory buffer; past experiences retrieved at inference |
| **ECHO** (2024) | Hindsight replay — generates counterfactual positive examples from failed trajectories |
| **ACE** (2025) | Builds persistent playbooks; learns from execution feedback purely via ICL |

And in production systems, this pattern is already emerging:

| System | How It Implements Trajectory Memory |
|--------|-------------------------------------|
| **Claude Code** | `CLAUDE.md` + `/memories` directory for cross-session skill accumulation |
| **Cursor** | CORE Memory MCP, codebase semantic search, Agent Skills |
| **Devin** | RAG as memory: codebase chunking + vector embedding |
| **Voyager** | Skill library with embedding-based retrieval |

The gap? All existing systems use **simple retrieval** — embed the task description, do kNN, done. Nobody is doing multi-level trajectory matching.

---

## The Hard Problem: Finding Similar Trajectories

A trajectory isn't a document. It's a structured, multi-step sequence with causal relationships:

```python
Trajectory = {
    task_description: str,          # What was the task?
    steps: [
        {
            thought: str,           # Reasoning process
            action: str,            # Executed action
            observation: str,       # Environment feedback
            reward: float,          # Optional reward signal
        }, ...
    ],
    final_answer: str,              # Final result
    metadata: {
        success: bool,
        domain: str,
        difficulty: float,
        key_skills: [str],          # Skills involved
    }
}
```

Existing papers use surprisingly simple retrieval — essentially "embed task description → kNN." ExpeL uses `all-mpnet-base-v2` with Faiss; Voyager embeds one-sentence skill descriptions and does cosine similarity for top-5. These work, but they leave a lot on the table.

### Multi-Level Embedding

The key insight: trajectories have **structure at multiple granularities**, and matching should happen at each level.

**Level 1 — Task Matching (coarse filtering):**
Embed `task_description`, cosine similarity for rapid candidate filtering. This is identical to standard RAG.

```python
user_query = "Write a Python function to compute the nth Fibonacci number"
# → embed(user_query) vs all trajectory task_emb
# → Match: "Implement recursive + memoized Fibonacci" (cosine=0.91)
```

**Level 2 — Strategy Matching (fine ranking):**
Summarize key decision points and reasoning strategies into text, embed, and compare. User queries typically lack explicit strategy info, so LLM inference bridges the gap:

```python
prompt = f"""Given this task: {user_query}
What problem-solving strategies might be needed?
Output as a brief summary."""
# → "Needs recursion or DP, consider memoization"
strategy_emb = embed(llm(prompt))
```

**Level 3 — Step Matching (fine-grained):**
Use DTW or soft-alignment to compare step sequences. The LLM decomposes the query into an expected step skeleton, then aligns it against stored trajectory steps.

The three levels combine into a weighted similarity score:

$$
\text{Sim}(T_1, T_2) = \alpha \cdot \cos(\mathbf{e}^1_{\text{task}}, \mathbf{e}^2_{\text{task}}) + \beta \cdot \cos(\mathbf{e}^1_{\text{strategy}}, \mathbf{e}^2_{\text{strategy}}) + \gamma \cdot \text{StepAlign}(S^1, S^2)
$$

where $\alpha$, $\beta$, $\gamma$ weight the contribution of task-level cosine similarity, strategy-level cosine similarity, and step-level alignment respectively.

> **Practical note:** In most scenarios, Level 1 alone is sufficient. Level 2 LLM query expansion offers the best cost/performance ratio. Level 3 is only needed for high-precision requirements like agent task orchestration.

### Alternative Retrieval Methods

Beyond multi-level embedding, several complementary approaches exist:

| Method | Training | Speed | Best For |
|--------|----------|-------|----------|
| **Multi-Level Embedding** | Low (existing encoders) | Fast | Primary recall + ranking |
| **Contrastive Learning** | High (needs data) | Fast | High-precision recall |
| **LLM-as-Judge** | None | Slow | Reranking top candidates |
| **Skill-Tag Retrieval** | Low | Fast | Coarse categorical filtering |

**Recommended pipeline:** Skill-Tag coarse filter → Multi-Level Embedding recall → LLM-as-Judge reranking.

---

## Building the Trajectory Database

We've established *how* to retrieve trajectories — but retrieved from what? A trajectory database needs to be populated first, and then it needs to grow. This is really one continuous problem, not two: the same mechanism that seeds the database on Day 1 is the mechanism that scales it to millions of entries.

### Bootstrapping: Getting the First Trajectories

In an enterprise setting, you can't start from generic benchmarks — you need trajectories that reflect proprietary toolchains, internal SOPs, and domain-specific workflows. The guiding principle: **deliver visible value without delay.**

**Phase 1 — Mine Existing Knowledge (Day 1–3):**
1. Analyze Jira / Slack / search logs to find the **Top 10 high-frequency tasks**
2. Extract workflows from SOPs / Confluence / Runbooks
3. LLM parses into structured task + step instructions
4. Strong model executes in the enterprise environment → verified trajectories stored

> Don't try to cover everything — just nail the Top 10 and deliver value on Day 1.

**Phase 2 — Capture Expert Behavior (Day 3–14):**
Observe enterprise experts' workflows in the background with zero disruption:

| Environment | Capture Method |
|-------------|---------------|
| Software Dev | IDE plugin — senior engineers' debug paths |
| Data Analysis | Jupyter notebook hooks — analysts' exploration trajectories |
| Customer Support | Ticket system integration — expert resolution sequences |
| DevOps | CLI wrapper / shell history — incident troubleshooting |

The key is **zero-effort capture** — experts don't change their workflow; the system learns by watching.

For general (non-enterprise) scenarios, you can bootstrap from existing benchmark trajectories (GSM8K CoT, SWE-bench patches), rejection sampling with a strong model, or AgentTrek-style tutorial synthesis.

### From Bootstrapping to Flywheel: Every Inference Is a Data Point

Here's where bootstrapping transitions into something more powerful. Once the system is serving real users, **every query becomes an opportunity to generate new trajectories** — not just answer the user.

The mechanism is surprisingly simple: on each query, sample N candidate trajectories in parallel (at varying temperatures). Return the best one to the user. Silently verify and store *all* successful ones in the background. The user sees one answer; the system learns from many.

```python
async def serve_and_accumulate(query, context, n_samples=8):
    # 1. Retrieve existing trajectories as ICL
    similar_trajs = trajectory_db.retrieve(query, top_k=3)
    
    # 2. Sample N trajectories in parallel (varying temperatures)
    trajectories = await asyncio.gather(*[
        model.solve(query, context, similar_trajs, 
                    temperature=t, trace=True)
        for t in [0.2, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0]
    ])
    
    # 3. Verify + rank
    verified = [(t, score) for t in trajectories 
                if (score := verify(t)) > THRESHOLD]
    
    # 4. Return best to user
    respond_to_user(max(verified, key=lambda x: x[1]))
    
    # 5. Silently store all successes (with dedup)
    for traj, score in verified:
        if should_store(traj, trajectory_db):
            trajectory_db.add(traj)
```

This is essentially **online rejection sampling** — the same technique used in offline RL data collection, but applied continuously during serving. The advantages over doing this offline are significant:

| Dimension | Offline (batch generation) | Online (during serving) |
|-----------|---------------------------|------------------------|
| Data source | Pre-collected benchmarks | Real user queries |
| Distribution match | ❌ May not align with needs | ✅ Perfectly matches usage |
| Continuity | One-time generation | Continuous growth |
| Additional cost | Dedicated GPU time | Piggybacking on inference compute |

**Cost control:** Not every query needs N samples. If the database already has highly similar trajectories, 1 sample suffices. For novel query types, sample more aggressively.

This creates the **flywheel**:

```
User Query → Retrieve Existing Trajectories → LLM Reasoning
    → Verify → ✅ Success → Quality Gate → Trajectory DB → Better Retrieval Next Time
              → ❌ Failure → Extract Reflection → Negative Example Store
```

More usage → more accumulated trajectories → more precise retrieval → better user experience → more usage. Unlike RL training which is done once and frozen, the trajectory database **continuously grows**.

### Quality Gate: Not Every Trajectory Is Worth Keeping

The flywheel only works if the database maintains high quality. A naive "store everything" approach leads to redundancy and noise.

```python
def should_store(trajectory, db):
    if not trajectory.verified_success:
        return False
    
    # Dedup: skip if too similar to existing
    most_similar = db.retrieve(trajectory.task, top_k=1)
    if most_similar and similarity(trajectory, most_similar[0]) > 0.95:
        return False
    
    # Novelty: prioritize new strategies or domains
    if compute_novelty(trajectory, db) < NOVELTY_THRESHOLD:
        return False
    
    # Quality: compress overly long trajectories
    if trajectory.num_steps > MAX_STEPS:
        trajectory = compress(trajectory)
    
    return True
```

In enterprise settings, add an admin review layer: tech leads tag trajectories as "team best practice," forming organizational knowledge assets rather than merely individual accumulation.

### Learning from Failures

Failed trajectories aren't discarded — they're mined for value:

1. **Reflection extraction:** LLM analyzes failure causes → stored as searchable metadata
2. **Counterfactual generation** (ECHO-style): Fix the failed trajectory → if the fixed version passes verification, store it
3. **Negative examples in context:** "Here's a failed attempt — avoid these mistakes"

### Verification Signals

What counts as "success" depends on the task:

| Task Type | Verification | Automation |
|-----------|-------------|------------|
| Math | Exact answer matching | ✅ Fully automated |
| Code | Unit tests pass | ✅ Fully automated |
| Agent tasks | Environment state check | ✅ Fully automated |
| Open QA | Implicit user feedback (👍/👎) | 🟡 Semi-automated |

---

## System Architecture

### Overall Pipeline

The system splits into offline indexing and online serving:

**Offline:**
1. Trajectory Generation → Quality Filtering → Structured Parsing
2. Multi-Level Embedding + Skill Tag Extraction
3. Index into Vector DB + Tag Index

**Online:**
1. User Query → Query Analysis
2. Skill Tag Matching → Embedding Retrieval → LLM Reranking
3. Context Assembly → LLM Inference with Retrieved Trajectories

### Trajectory Database Design

```python
class TrajectoryDB:
    def __init__(self):
        self.vector_store = VectorDB()       # FAISS / Milvus / Qdrant
        self.tag_index = InvertedIndex()      # skill tag → trajectory IDs
        self.trajectory_store = DocumentDB()  # Full trajectory storage
    
    def retrieve(self, query: str, top_k: int = 5):
        # Stage 1: Skill-tag filtering
        query_tags = extract_skill_tags_from_query(query)
        candidates = self.tag_index.search(query_tags, top_n=50)
        
        # Stage 2: Multi-level embedding ranking
        query_emb = embed(query)
        ranked = self.vector_store.search(
            query_emb, candidates=candidates, top_k=10
        )
        
        # Stage 3: LLM reranking
        trajectories = [self.trajectory_store.get(id) for id in ranked]
        return llm_rerank(query, trajectories, top_k=top_k)
```

### Context Assembly

Retrieved trajectories must be **compressed** before entering the context — raw trajectories can be thousands of tokens. Extract key decision points rather than the full trace:

```
System: You are solving a task. Here are similar solved problems for reference.

=== Similar Trajectory 1 (similarity: 0.92) ===
Task: {similar_task_1}
Solution approach:
  Step 1: {thought_1} → {action_1} → {observation_1}
  Step 2: {thought_2} → {action_2} → {observation_2}
Final answer: {answer_1}

=== Your Task ===
Task: {user_query}
Please solve this step by step, referencing the approaches above when helpful.
```

---

## Does It Actually Work? Risks and Evidence

> This is the greatest risk to the approach. If LLMs cannot effectively utilize trajectories in context, the entire system falls apart.

### What the Research Says

| Finding | Source | Implication |
|---------|--------|-------------|
| **Lost in the Middle:** LLMs best use info at context boundaries; middle content is ignored | Liu et al., 2023 | 🔴 Position trajectories carefully |
| **Many-Shot ICL works:** Hundreds of demos yield continuous improvement following power-law | DeepMind, 2024 | 🟢 More trajectories help |
| **One great demo > many mediocre ones** | ACL 2024 | 🟢 Retrieval quality > quantity |
| **Longer CoT steps improve reasoning** (even without new info) | arXiv 2024 | 🟢 Full trajectories may beat summaries |
| **Model-generated rationales can substitute for human-annotated ones** | DeepMind, 2024 | 🟢 Trajectories can be self-generated |

### Mitigating the Risks

**Lost in the Middle:** Place trajectories at the end of context, immediately before the user query. Compress to key decision points (3–5 steps, not 20). Front-load strategy summaries.

**Context Length Pressure:** Use hierarchical presentation — summary first, key steps next, full trajectory only if budget allows. Dynamically allocate context based on query complexity.

**Inter-Trajectory Interference:** Limit to top 1–2 trajectories. If multiple strategies are presented, explicitly annotate differences. Let the model self-select via MCTS-style browsing.

**Format Mismatch:** Use the same model to generate trajectories (self-play ensures consistent reasoning style). Standardize trajectory format to match the target model's CoT conventions.

### Hypotheses That Need Testing

| Hypothesis | How to Validate |
|------------|----------------|
| 1 quality trajectory > 3 mediocre ones | Ablation: 1 vs 3 vs 5 trajectories |
| Compressed summary vs full trajectory | Compare full / summary / key-steps-only |
| Trajectory position matters | Beginning vs middle vs end of context |
| Same-model vs cross-model trajectories | Cross-model transfer experiments |
| Trajectory ICL ≈ RL with same data | Fair comparison controlling trajectory count |

---

## Experiment Plan

### Baselines

| Experiment | Description |
|------------|-------------|
| **Base LLM** | No trajectory assistance |
| **Few-shot ICL** | Random examples as demos |
| **RAG-standard** | Retrieved documents (not trajectories) |
| **Trajectory ICL** (ours) | Retrieved structured trajectories |
| **RL Post-trained** | GRPO training on the same trajectory data |

### Benchmarks

- **Math:** GSM8K, MATH, AIME
- **Code:** HumanEval, MBPP, SWE-bench
- **Agent:** WebArena, ALFWorld, OSWorld

### Metrics

- Task success rate (pass@1 on verifiable tasks)
- Retrieval quality (human-judged relevance)
- Efficiency (GPU hours vs retrieval latency)
- Scaling curve (performance vs database size)
- Generalization (cross-domain transfer)

---

## Open Questions

1. **Trajectory Compression:** How aggressively can you compress without losing key decision signals?

2. **Negative Trajectories:** How to best use failures — "avoid this" warnings, ECHO-style counterfactuals, or Reflexion-style abstractions?

3. **Dynamic Retrieval:** Can you retrieve different trajectories at different *steps* of reasoning, not just at the beginning?

4. **Quality vs Diversity:** The most similar trajectory may not inspire the most creative solution. When does moderate dissimilarity help?

5. **Complementarity with RL:** The most promising direction may be a hybrid — use Trajectory ICL for rapid bootstrapping, then apply RL fine-tuning on the most valuable trajectories. A "retrieval-first, training-supplementary" paradigm.

---

## References

### Agent Experiential Learning
1. **ExpeL** — Zhao et al., 2023 — Autonomous experience collection + retrieval-augmented ICL
2. **Voyager** — Wang et al., 2023 — Skill library with embedding-based retrieval for Minecraft
3. **Reflexion** — Shinn et al., 2023 — Verbal reflections from failed trajectories as episodic memory
4. **CER** (Contextual Experience Replay) — 2024 — Training-free dynamic memory buffer
5. **ECHO** — 2024 — Hindsight experience replay with counterfactual trajectory generation
6. **ACE** (Agentic Context Engineering) — 2025 — Persistent playbooks and ICL-based learning

### Trajectory Synthesis and Utilization
7. **AgentTrek** (ICLR 2025 Spotlight) — Automated GUI agent trajectory synthesis from web tutorials
8. **TrajICL** — 2024 — Trajectory prediction via ICL with spatio-temporal similarity
9. **STEP** — 2025 — Step-level trajectory decomposition for fine-grained credit assignment

### Retrieval-Augmented Reasoning
10. **RAT** (Retrieval Augmented Thoughts) — 2024 — RAG + CoT synergy
11. **R3-RAG** — 2024 — RL-trained optimal retrieval and reasoning trajectory
12. **RAS** — 2026 — Dynamic question-specific knowledge graph construction

### Context Utilization in LLMs
13. **Lost in the Middle** — Liu et al., 2023 — U-shaped attention curve in long contexts
14. **Many-Shot ICL** — Google DeepMind, 2024 — Power-law improvement with demo count
15. **Reinforced ICL** — DeepMind, 2024 — Model-generated rationales as effective substitutes

*Code examples are synthesized implementations illustrating practical patterns.*
