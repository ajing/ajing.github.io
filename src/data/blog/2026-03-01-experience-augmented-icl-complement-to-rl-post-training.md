---
author: Jing Lu
pubDatetime: 2026-03-01T00:00:00Z
title: "Experience-Augmented In-Context Learning: A Training-Free Complement to RL Post-Training"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Agents
  - RAG
description: "RL post-training makes models smarter, but it can't cover the infinite long tail of real-world cases. Experience-augmented ICL retrieves successful reasoning traces at inference time, letting agents learn continuously from real usage — no retraining required."
---

**RL post-training works.** GRPO and PPO can teach models to reason, use tools, and recover from errors. But it's a one-time investment — significant GPU hours, careful reward engineering, and the resulting policy is **frozen at training time**. You can RL-train for math, coding, and general tool use. You can't RL-train for every enterprise workflow, every niche domain, every edge case your agent will encounter in the wild.

The real world has an infinite long tail. Agents need a way to keep learning from what actually happens on the ground.

This post proposes **Experience-Augmented ICL**: store successful reasoning trajectories in an external database, retrieve the most relevant ones at inference time, and guide the model through in-context learning — no parameter updates required. It uses the **same trajectory data** that RL trains on, but at inference time through retrieval rather than weight updates. RL bakes patterns into weights; experience retrieval keeps them external and continuously growing.

---

## Positioning: Where Experience-Augmented ICL Fits

To frame this precisely, consider the landscape of methods for improving LLM reasoning *after* pretraining:

| Method | Parameter Update | Compute Profile | Adaptation Scope |
|--------|-----------------|-----------------|------------------|
| **RL Post-Training** (GRPO/PPO) | ✅ Full / LoRA | GPU hours (offline) | Global policy shift |
| **Test-Time Training** (TTT) | ✅ Temporary per-query | Medium (per query) | Single-instance adaptation |
| **Test-Time Compute Scaling** (best-of-N, MCTS) | ❌ None | High inference (per query) | Single-instance search |
| **Standard RAG** | ❌ None | Retrieval + inference | Knowledge augmentation |
| **Experience-Augmented ICL** (this work) | ❌ None | Retrieval + inference | Strategy augmentation |

The key distinction from standard RAG: we're not retrieving *knowledge* (facts, documents), we're retrieving *strategies* — structured sequences of reasoning steps that demonstrate how to solve similar problems. And unlike test-time compute scaling (which burns compute searching over solutions to the current problem), experience-augmented ICL amortizes that search cost across problems: the successful searches from past problems become the demonstrations for future ones.

> **Connection to test-time compute scaling:** Best-of-N sampling and experience-augmented ICL are complementary. Best-of-N searches the solution space for the *current* query. Experience retrieval biases that search using solutions to *past* queries. The combination is especially powerful: use retrieved trajectories to guide sampling, then run best-of-N within that guided distribution. This is effectively **amortized test-time compute** — the expensive search from previous queries reduces the search needed for new ones.

---

## Related Work

Several research threads converge on this idea, though none fully address multi-level trajectory retrieval:

**Agent Experiential Learning:**

| Work | Approach | Retrieval Method | Key Limitation |
|------|----------|-----------------|----------------|
| **ExpeL** (Zhao et al., 2023) | Collects success/failure experiences; retrieves via ICL | `all-mpnet-base-v2` + Faiss kNN on task description | Single-level retrieval only |
| **Voyager** (Wang et al., 2023) | Skill library for Minecraft agent | OpenAI embedding on one-sentence skill descriptions | Retrieves descriptions, not full trajectories |
| **Reflexion** (Shinn et al., 2023) | Verbal reflections from failures as episodic memory | Recency-based (most recent reflection) | No similarity-based retrieval |
| **CER** (2024) | Training-free dynamic memory buffer | Associative retrieval on context | Compressed experience summaries only |
| **ECHO** (2024) | Hindsight replay — counterfactual positives from failures | N/A (augmentation, not retrieval) | Trajectory generation, not retrieval |
| **ACE** (2025) | Persistent playbooks; ICL-based learning from feedback | Rule-based playbook matching | No embedding-based similarity |

**Key gap:** All existing work performs single-level retrieval — embed the task description, run kNN, return results. Nobody attempts strategy-level or step-level matching, which is where the interesting retrieval challenges lie.

**Trajectory Synthesis:**
AgentTrek (ICLR 2025 Spotlight) automatically synthesizes GUI agent trajectories from web tutorials. STEP (2025) decomposes trajectories at the step level for fine-grained credit assignment. These are complementary: they address *how to generate* trajectories, while we address *how to retrieve and use* them.

**Production systems** are converging on this pattern independently: Claude Code uses `CLAUDE.md` + `/memories` for cross-session accumulation; Cursor has CORE Memory MCP and Agent Skills; Devin uses RAG over codebase chunks. All use simple retrieval — the retrieval quality problem remains unsolved.

---

## Trajectory Retrieval: The Core Technical Challenge

### Why Simple Retrieval Falls Short

A trajectory is not a document. It's a structured, multi-step sequence with causal dependencies between steps:

```python
Trajectory = {
    task_description: str,
    steps: [
        { thought: str, action: str, observation: str, reward: float },
        ...
    ],
    final_answer: str,
    metadata: { success: bool, domain: str, key_skills: [str] }
}
```

Standard embedding-based retrieval treats this as a flat text blob — embed the task description, compute cosine similarity, return top-K. This works for *task-level* similarity ("find Fibonacci" matches "implement Fibonacci") but misses deeper structure:

- Two trajectories solving the same task type with **different strategies** (DP vs. greedy) will have high task-level similarity but low strategy-level similarity.
- Two trajectories in **different domains** that use the **same reasoning pattern** (e.g., divide-and-conquer in both algorithm design and system debugging) will have low task-level similarity but high strategy-level similarity.

### Multi-Level Similarity

We propose matching at three granularities:

**Level 1 — Task Embedding (coarse filter, standard RAG):**
Embed `task_description`, cosine similarity. This is the baseline that all existing papers use.

**Level 2 — Strategy Embedding (the key differentiator):**
Summarize each trajectory's key decisions and reasoning strategy into a condensed text representation, then embed. This captures *how* the problem was solved, not just *what* was solved.

The asymmetric retrieval challenge: user queries contain task information but rarely state their strategy needs explicitly. Two approaches to bridge this gap:

```python
# Option A: LLM-based query expansion
strategy_query = llm(f"What problem-solving strategies might be needed for: {query}")
strategy_emb = embed(strategy_query)

# Option B: Dual encoder trained on (query, strategy_summary) pairs
strategy_emb = strategy_encoder(query)  # Learned projection into strategy space
```

Option A adds ~100ms latency but requires no training. Option B is faster at inference but requires paired training data (queries matched to successful strategy summaries).

**Level 3 — Step-Level Alignment (fine-grained, expensive):**
Compare the step sequences of two trajectories using soft-DTW or optimal transport alignment. For query-to-trajectory matching, the LLM first decomposes the query into an expected step skeleton.

The combined similarity:

$$
\text{Sim}(T_1, T_2) = \alpha \cdot \cos(\mathbf{e}^1_{\text{task}}, \mathbf{e}^2_{\text{task}}) + \beta \cdot \cos(\mathbf{e}^1_{\text{strategy}}, \mathbf{e}^2_{\text{strategy}}) + \gamma \cdot \text{StepAlign}(S^1, S^2)
$$

> **When to use each level:** Level 1 alone is sufficient for ~70% of use cases (same domain, similar problems). Level 2 matters when the trajectory database spans multiple domains or when multiple solution strategies exist for the same problem type. Level 3 is only justified for high-stakes agent orchestration tasks where step-level plan similarity predicts execution success.

### Embedding Model Selection

The choice of embedding model matters more than the architecture above it:

| Model Category | Examples | Trade-off |
|---------------|----------|-----------|
| **General sentence encoders** | `all-mpnet-base-v2`, `e5-large-v2` | Off-the-shelf, no training; doesn't understand trajectory structure |
| **Code+text bimodal** | `voyage-code-3` | Understands code snippets in trajectories; strong for coding agent tasks |
| **Instruction-tuned** | `gte-Qwen2`, `SFR-Embedding-2` | Better at capturing semantic differences like "DP approach" vs "greedy approach" |
| **Fine-tuned contrastive** | Trained on trajectory pairs | Best precision but requires trajectory-pair training data |

**Practical observation:** ExpeL's `all-mpnet-base-v2` baseline is hard to beat for task-level retrieval. The gains from better embeddings show up primarily at Level 2 (strategy matching), where instruction-tuned models like `gte-Qwen2` capture reasoning-style differences that general encoders miss.

### Retrieval Pipeline

For a production system, multi-stage retrieval is essential:

| Stage | Method | Output | Purpose |
|-------|--------|--------|---------|
| Coarse filter | Skill-tag matching (inverted index) | ~50 candidates | Eliminate obviously irrelevant trajectories |
| Recall | Multi-level embedding search (ANN) | ~10 candidates | Semantic ranking across task, strategy, and step levels |
| Rerank | LLM-as-Judge or cross-encoder | ~3 candidates | Fine-grained relevance judgment |

The LLM reranking stage is expensive but high-value: given the query and 10 candidate trajectory summaries, the LLM selects the most relevant 2–3. This is where the biggest quality gains come from — the LLM can reason about subtle relevance factors that embedding similarity misses.

### Multi-Turn Trajectories: The Evolving Query Problem

Everything above assumes a one-shot setting: a user query arrives, we retrieve trajectories, done. But most real agent interactions are **multi-turn** — the user's intent evolves, intermediate results change the trajectory, and what's "similar" shifts at each step. This introduces three distinct challenges.

**Challenge 1: What do we embed on the query side?**

At turn *t*, the user has exchanged *t* messages with the agent. The naive approach — embed only the latest user message — fails for the same reason it fails in conversational search: "now debug it" is meaningless without context. But concatenating the entire conversation history produces an embedding dominated by earlier turns, diluting the current intent.

Four approaches, in order of increasing sophistication:

**Approach A — Sliding Window Concatenation:**
Concatenate the last *k* turns (typically *k* = 3–5) and embed the result. Simple, no LLM call needed, but the embedding quality degrades as the window includes irrelevant earlier context.

```python
def embed_multiturn_query(history: List[Turn], k: int = 3):
    window = history[-k:]
    text = "\n".join([f"{t.role}: {t.content}" for t in window])
    return embed(text)
```

**Approach B — LLM Query Rewriting (TREC CAsT standard):**
Use an LLM to rewrite the current turn into a self-contained query that incorporates necessary context. This is the approach validated by the conversational search community and is the most robust for retrieval quality.

```python
def rewrite_for_retrieval(current_turn: str, history: List[Turn]):
    prompt = f"""Given this conversation:
{format_history(history)}

Rewrite the latest message to be self-contained: "{current_turn}"
"""
    return llm(prompt)
    # "now debug it" → "debug the sorting function from the previous implementation"
```

**Which LLM for rewriting?** Query rewriting is fundamentally a **simple NLU task** — coreference resolution ("it" → "the sorting function") and context incorporation ("debug" → "debug the Python implementation from the previous step"). This is closer to T5-level seq2seq than frontier-model reasoning. TREC CAsT participants typically used fine-tuned T5-base/T5-large, and these consistently outperformed zero-shot large models for this specific task.

| Model Tier | Examples | Latency | When to Use |
|-----------|----------|---------|-------------|
| **Fine-tuned small** (2B–8B) | Gemma 2B, Phi-3-mini, T5-large | 10–30ms | Production — best latency, sufficient quality for most cases |
| **Instruction-tuned medium** | Llama 3.1 8B, Gemini Flash | 30–80ms | When you lack fine-tuning data; zero-shot rewriting |
| **Frontier** | GPT-4o, Claude Sonnet | 200–500ms | Complex multi-hop intent resolution; or as teacher for distillation |

The **best production pattern** is teacher-student distillation: use a frontier model offline to generate (conversation, rewritten_query) training pairs, then fine-tune a small 2B–8B model that serves at 10–30ms. A few thousand rewriting examples are typically sufficient — the task's low complexity means small models learn it quickly.

ConvDR (Yu et al., 2021) showed that learned rewriting consistently outperforms concatenation approaches across conversational retrieval benchmarks.

**Approach C — Hierarchical Pooling:**
Embed each turn independently, then combine using a weighted pooling scheme. This preserves turn-level structure and allows recency weighting:

$$
\mathbf{e}_{\text{query}} = \sum_{i=1}^{t} w_i \cdot \text{embed}(\text{turn}_i), \quad w_i = \frac{e^{\lambda i}}{\sum_j e^{\lambda j}}
$$

where λ > 0 controls recency bias. Higher λ puts more weight on recent turns. This is computationally cheap (turn embeddings can be cached and incrementally updated) and avoids the information loss of truncation, but the linear combination of embeddings loses compositional semantics — "cancel the order" and "order a cancellation" would produce similar pooled embeddings despite very different intents.

**Approach D — Turn-Aware Contrastive Encoder:**
Train a dedicated encoder on (multi-turn conversation, relevant trajectory) pairs. The encoder learns to project a conversation prefix into the same space as complete trajectory representations. This is the highest-quality approach but requires training data that's expensive to collect.

> **Practical recommendation:** Start with Approach B (LLM rewriting). It's the best cost/quality trade-off and doesn't require training data. Move to Approach D only if retrieval precision is the bottleneck and you have sufficient paired data. Approach C is attractive for latency-sensitive settings where you can't afford the rewriting LLM call.

**Challenge 2: How do we embed multi-turn trajectories on the storage side?**

Stored trajectories are themselves multi-step sequences. The question is what unit to embed and index:

| Granularity | What Gets Embedded | Retrieval Behavior |
|-------------|-------------------|-------------------|
| **Whole trajectory** | Full task description + all steps | Matches on overall task similarity; misses partial overlaps |
| **Per-step** | Each (thought, action, observation) triple | Can match mid-trajectory; high index size |
| **Sub-trajectory windows** | Overlapping windows of *k* consecutive steps | Balances granularity and index size |
| **Hierarchical** | Task-level + strategy-level + step-level (our multi-level approach) | Best coverage; highest engineering complexity |

For multi-turn retrieval specifically, **sub-trajectory windowing** is valuable: if the user is at step 5 of a 15-step task, you want to retrieve trajectories that had a similar step 5, not just a similar starting task. This is the partial trajectory matching problem — matching an in-progress trajectory against completed ones.

```python
def index_trajectory_windows(trajectory, window_size=3, stride=1):
    """Index overlapping sub-trajectory windows for step-level matching."""
    windows = []
    for i in range(0, len(trajectory.steps) - window_size + 1, stride):
        window = trajectory.steps[i:i + window_size]
        window_text = " → ".join([
            f"[{s.action}] {s.thought}" for s in window
        ])
        windows.append({
            "embedding": embed(window_text),
            "trajectory_id": trajectory.id,
            "start_step": i,
            "context": trajectory.task_description
        })
    return windows
```

**Challenge 3: When to re-retrieve?**

In a single-turn setting, you retrieve once. In multi-turn, the relevance of retrieved trajectories changes as the conversation evolves. Two strategies:

- **Retrieve once, at conversation start.** Simple, but the initial retrieval may become irrelevant by turn 5. Works when the task is well-defined from the start.
- **Re-retrieve at each turn (or at key decision points).** Expensive, but allows the system to adapt — if the user pivots from "implement a sort" to "now benchmark it against numpy," the system retrieves trajectories relevant to *benchmarking*, not sorting. The re-retrieval can be triggered selectively: re-retrieve when the cosine similarity between the current turn embedding and the original query embedding drops below a threshold, signaling a topic shift.

```python
async def multiturn_serve(conversation: List[Turn], trajectory_db):
    # Initial retrieval
    initial_query = rewrite_for_retrieval(conversation[-1].content, conversation)
    initial_emb = embed(initial_query)
    retrieved = trajectory_db.retrieve(initial_query, top_k=3)
    
    for new_turn in incoming_turns():
        conversation.append(new_turn)
        current_query = rewrite_for_retrieval(new_turn.content, conversation)
        current_emb = embed(current_query)
        
        # Re-retrieve if intent has shifted significantly
        if cosine_similarity(current_emb, initial_emb) < DRIFT_THRESHOLD:
            retrieved = trajectory_db.retrieve(current_query, top_k=3)
            initial_emb = current_emb  # Reset anchor
        
        response = model.generate(conversation, retrieved_trajectories=retrieved)
        yield response
```

> **The fundamental tension:** More frequent re-retrieval improves relevance but adds latency and can cause jarring context switches (the model was following Trajectory A's strategy, then suddenly gets Trajectory B). A middle ground: re-retrieve but bias toward trajectories consistent with the strategy already in progress.

---

## Trajectory Generation and the Accumulation Flywheel

### Bootstrapping

Initial trajectories can come from three sources:

1. **Rejection sampling with a strong model:** Sample N solutions per task, filter by verification (unit tests, exact match, environment checks). This is the same data pipeline used for RL post-training — the difference is we *store* the trajectories instead of training on them.

2. **Extraction from existing benchmarks:** GSM8K already has chain-of-thought traces. SWE-bench has patches. WebArena has interaction logs. Convert these to the trajectory format.

3. **Expert behavior capture:** Instrument expert workflows (IDE plugins, Jupyter hooks, CLI wrappers) to record real problem-solving trajectories with zero extra effort from experts.

### Online Accumulation: Every Inference Is a Data Point

Here's where bootstrapping transitions into something more powerful. Once the system is serving real queries, **every inference becomes an opportunity to generate new trajectories** — not just answer the user.

The mechanism: on each query, sample N candidate trajectories in parallel at varying temperatures. Return the best to the user. Verify all of them in the background. Store every verified success.

```python
async def serve_and_accumulate(query, context, n_samples=8):
    similar_trajs = trajectory_db.retrieve(query, top_k=3)
    
    trajectories = await asyncio.gather(*[
        model.solve(query, context, similar_trajs, temperature=t, trace=True)
        for t in [0.2, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0]
    ])
    
    verified = [(t, score) for t in trajectories 
                if (score := verify(t)) > THRESHOLD]
    
    respond_to_user(max(verified, key=lambda x: x[1]))
    
    for traj, score in verified:
        if should_store(traj, trajectory_db):
            trajectory_db.add(traj)
```

This is online rejection sampling — the same technique used for offline RL data collection, but applied continuously during serving. The key advantages:

- **Distribution match:** Offline rejection sampling operates on pre-collected benchmarks that may not reflect real usage patterns. Online accumulation perfectly matches the actual query distribution.
- **Amortized cost:** The N samples piggyback on inference compute that's already being spent. The marginal cost of storing verified trajectories is negligible.
- **Self-improvement:** Retrieved trajectories improve the sampling distribution for future queries. As the database grows, each subsequent query requires fewer samples to find a good solution.

**Cost control matters:** Not every query needs N samples. The sampling budget should be dynamic — if the database already contains highly similar trajectories (cosine > 0.9), one sample suffices. For novel query types, sample aggressively.

### Quality Gate

The flywheel only works if the database maintains high signal-to-noise ratio:

```python
def should_store(trajectory, db):
    if not trajectory.verified_success:
        return False
    
    # Dedup: skip if too similar to existing entries
    most_similar = db.retrieve(trajectory.task, top_k=1)
    if most_similar and similarity(trajectory, most_similar[0]) > 0.95:
        return False
    
    # Novelty: prioritize trajectories that use new strategies or cover new domains
    if compute_novelty(trajectory, db) < NOVELTY_THRESHOLD:
        return False
    
    # Compress overly long trajectories
    if trajectory.num_steps > MAX_STEPS:
        trajectory = compress(trajectory)
    
    return True
```

**Leveraging failures:** Failed trajectories have value too. Reflexion-style reflection extraction ("Step 3 chose greedy when DP was needed") can be stored as searchable metadata. ECHO-style counterfactual generation can fix failed trajectories — if the fixed version verifies, store it. Failed trajectories can also serve as negative examples in context: "Here's an approach that failed — avoid this pattern."

---

## Context Engineering: Making Trajectories Useful In-Context

Retrieving the right trajectory is necessary but not sufficient. The model must actually *use* it effectively. This section addresses the non-obvious challenges.

### The Lost-in-the-Middle Problem

Liu et al. (2023) showed that LLMs exhibit a U-shaped attention curve: information at the beginning and end of context is utilized well; middle content is largely ignored. Trajectories are long (easily 500–2000 tokens each), and placing multiple trajectories in context means key decision steps often land in the dead zone.

**Mitigations that work:**
- **Position control:** Place trajectories at the end of context, immediately before the user query. This exploits the recency bias in attention.
- **Front-load strategy summaries:** Begin each trajectory with a one-line "Core strategy: X" header. Even if detailed steps are ignored, the strategy signal reaches the model.
- **Hierarchical compression:** Present a 50-token strategy summary first, then 200-token key steps, then full trajectory only if context budget allows.

### Trajectory Compression: The Token Efficiency Problem

Raw trajectories are expensive. A 15-step agent trajectory with full thought/action/observation at each step easily consumes 3000+ tokens. With 3 retrieved trajectories, that's 9000+ tokens — a significant fraction of the context budget that could be used for the actual task.

The compression question has an information-theoretic framing: **what is the minimum description of a trajectory that preserves the decision-relevant information?**

Three compression strategies, ordered by aggressiveness:

| Strategy | Token Budget | What's Preserved | What's Lost |
|----------|-------------|-------------------|-------------|
| **Full trajectory** | 1000–3000 per traj | Everything | Nothing |
| **Key steps only** | 200–500 per traj | Decision points, strategy pivots | Routine steps, verbose observations |
| **Strategy summary** | 50–150 per traj | High-level approach, key insight | Step-level detail, execution specifics |

An interesting finding from the reasoning step length literature (arXiv 2024): lengthening CoT steps improves reasoning even when no new information is added. This suggests that for reasoning tasks, **some verbosity in trajectories may actually help** — the detailed step-by-step format itself provides a scaffolding effect. The optimal compression level likely depends on task type: math/code benefits from detailed steps; agent tasks benefit more from strategy summaries.

### Inter-Trajectory Interference

When multiple retrieved trajectories employ contradictory strategies, the model may become confused. If Trajectory 1 solves a problem with DP and Trajectory 2 uses greedy, naively presenting both can degrade performance below the single-trajectory baseline.

**Mitigations:**
- Retrieve top 1–2 trajectories only (quality > quantity — confirmed by ACL 2024 findings showing one carefully selected demo can outperform multiple)
- When presenting multiple trajectories, explicitly annotate differences and when each approach is preferred
- **Self-selection:** Have the model browse trajectory summaries first and select the most relevant one before seeing full details (analogous to MCTS node selection)

### Format Alignment

Trajectory format matters more than expected. If the retrieved trajectory uses a different reasoning style than what the model naturally produces (e.g., different CoT notation, different action formats), the trajectory can be counterproductive.

**Best practice:** Generate trajectories using the same model that will consume them (self-play). This ensures the trajectory's reasoning style, vocabulary, and format naturally match the model's own patterns. Cross-model trajectory transfer (e.g., using GPT-4 trajectories with Llama) works but requires format normalization.

---

## When Experience-Augmented ICL Falls Short

This approach is not universally superior to RL post-training. It's important to be precise about where it breaks down:

**1. Distribution shift in reasoning style.** RL post-training *changes how the model reasons* — it can learn to prefer certain tool-calling patterns, develop new heuristics, or suppress failure modes. Experience-augmented ICL can only show the model what good reasoning looks like; it can't reshape the model's inherent biases. For tasks where the base model has deeply ingrained bad habits, ICL demonstrations may not override them.

**2. Context budget constraints.** Experience-augmented ICL trades parameter storage for context storage. For complex tasks requiring 5+ reference trajectories of 1000+ tokens each, the context budget for actual reasoning shrinks. RL post-training encodes patterns in weights — zero context overhead.

**3. Latency sensitivity.** The retrieval pipeline (embedding + ANN search + optional LLM reranking) adds 50–200ms. For latency-critical applications, this overhead may be unacceptable. RL post-training has zero inference overhead.

**4. Novelty gap.** If the trajectory database has no similar entries for a truly novel query, retrieval returns irrelevant trajectories that can *hurt* performance. RL post-training generalizes through learned policy; experience retrieval requires explicit coverage.

**Where experience-augmented ICL wins:** rapid deployment (hours vs. days), continuous improvement without retraining, domain adaptation without catastrophic forgetting, and the ability to maintain multiple strategy libraries for different user populations.

> **The hybrid hypothesis:** The most promising direction may not be either/or. Use experience-augmented ICL for rapid bootstrapping and long-tail coverage, then apply RL fine-tuning on the highest-value trajectory patterns. The trajectory database tells you *what* to train on; RL training tells the model to internalize it. This "retrieval-first, training-supplementary" paradigm is the natural next step.

---

## Experimental Design

### Controlled Comparison

The critical experiment: **does experience-augmented ICL match RL post-training when given access to the same trajectories?**

| Condition | Description |
|-----------|-------------|
| Base LLM | No assistance |
| Random few-shot ICL | Randomly selected demonstrations |
| Standard RAG | Retrieved documents (not trajectories) |
| Experience ICL (Level 1) | Task-level retrieval only |
| Experience ICL (Level 1+2) | Task + strategy retrieval |
| Experience ICL (Level 1+2+3) | Full multi-level retrieval |
| RL post-trained (same data) | GRPO/PPO trained on the same trajectory set |

### Key Ablations

| Variable | Conditions | Tests |
|----------|-----------|-------|
| Number of trajectories | 1, 3, 5, 10 | Quality vs. quantity trade-off |
| Compression level | Full / key-steps / summary | Token efficiency vs. information preservation |
| Context position | Beginning / middle / end | Lost-in-the-middle effect |
| Cross-model transfer | Same model / different model | Self-play vs. cross-model trajectories |
| Database size scaling | 100 / 1K / 10K / 100K entries | Scaling behavior and diminishing returns |

### Benchmarks

- **Math reasoning:** GSM8K, MATH, AIME (verifiable, exact match)
- **Code generation:** HumanEval, MBPP, SWE-bench (verifiable, test-based)
- **Agent tasks:** WebArena, ALFWorld, OSWorld (verifiable, environment-based)

### Metrics

- **pass@1** on verifiable tasks (primary)
- **Retrieval precision@K** — are retrieved trajectories actually relevant? (human eval)
- **Token efficiency** — task performance per context token consumed
- **Scaling exponent** — how does performance scale with log(database size)?
- **Cross-domain transfer** — do math trajectories help code? Do code trajectories help agent tasks?

---

## Open Research Questions

1. **Optimal compression:** What is the minimum trajectory representation that preserves decision-relevant information? Can we learn a task-dependent compression function?

2. **Dynamic retrieval during reasoning:** Current design retrieves trajectories once before generation. Can we re-retrieve *mid-reasoning* — e.g., when the model hits a dead end at step 5, retrieve step-level similar partial trajectories?

3. **Quality vs. diversity:** The most similar trajectory maximizes relevance but may bias the model toward one solution path. Deliberately introducing moderately dissimilar trajectories (MMR-style) could improve exploration. When does diversity help?

4. **Negative trajectory utilization:** What's the optimal way to present failure information? "Avoid this approach" ICL, ECHO-style counterfactual correction, or Reflexion-style abstract lessons?

5. **Trajectory database scaling laws:** Does performance follow a power law with database size (as many-shot ICL suggests), and if so, what's the exponent? At what scale do diminishing returns dominate?

6. **Hybrid ICL + RL training:** Use the trajectory database to identify which patterns to distill into the model via RL. The database acts as a curriculum — high-retrieval-frequency trajectories signal which skills the model should internalize.

---

## References

### Agent Experiential Learning
1. **ExpeL** — Zhao et al., 2023 — Autonomous experience collection + retrieval-augmented ICL
2. **Voyager** — Wang et al., 2023 — Skill library with embedding-based retrieval for Minecraft
3. **Reflexion** — Shinn et al., 2023 — Verbal reflections from failed trajectories as episodic memory
4. **CER** (Contextual Experience Replay) — 2024 — Training-free dynamic memory buffer
5. **ECHO** — 2024 — Hindsight experience replay with counterfactual trajectory generation
6. **ACE** (Agentic Context Engineering) — 2025 — Persistent playbooks and ICL-based learning

### Trajectory Synthesis
7. **AgentTrek** (ICLR 2025 Spotlight) — Automated GUI agent trajectory synthesis from web tutorials
8. **TrajICL** — 2024 — Trajectory prediction via ICL with spatio-temporal similarity (step-level matching)
9. **STEP** — 2025 — Step-level trajectory decomposition for fine-grained credit assignment
10. **RLEP** — 2024 — Experience replay to accelerate RL reasoning training

### Retrieval-Augmented Reasoning
11. **RAT** (Retrieval Augmented Thoughts) — 2024 — RAG + CoT synergy for multi-step reasoning
12. **R3-RAG** — 2024 — RL-trained optimal retrieval + reasoning trajectory
13. **RAS** — 2026 — Dynamic question-specific knowledge graph construction

### Context Utilization and Test-Time Compute
14. **Lost in the Middle** — Liu et al., 2023 — U-shaped attention curve in long contexts
15. **Many-Shot ICL** — Google DeepMind, 2024 — Power-law improvement with demonstration count
16. **Reinforced ICL** — DeepMind, 2024 — Model-generated rationales as effective substitutes
17. **Scaling LLM Test-Time Compute** — Snell et al., 2024 — Optimal compute allocation at inference
18. **Reasoning Step Length** — arXiv 2024 — Longer CoT steps improve reasoning independent of information content

### Multi-Turn Conversational Retrieval
19. **ConvDR** — Yu et al., 2021 (arXiv:2104.13650) — Few-shot conversational dense retrieval with history encoding
20. **TREC CAsT** — Conversational Assistance Track — Benchmark for conversational search with query rewriting

*Code examples are synthesized implementations illustrating practical patterns.*
