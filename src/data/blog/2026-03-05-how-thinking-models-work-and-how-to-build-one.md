---
author: Jing Lu
pubDatetime: 2026-03-05T00:00:00Z
title: "How Thinking Models Work and How to Build One"
featured: true
draft: true
tags:
  - AI
  - LLM
  - ML Engineering
  - Reasoning
  - Post-Training
description: "Thinking models like o1 and DeepSeek-R1 produce extended chains of internal reasoning before answering. This post breaks down how they work, where reasoning comes from across pretraining, post-training, and inference, and the types of data that make thinking emerge."
---

The biggest shift in LLM capability since GPT-4 hasn't been a new architecture or a bigger pretraining corpus — it's been **teaching models to think before they speak**.

Models like OpenAI o1/o3, DeepSeek-R1, and Claude 3.7 Sonnet with extended thinking don't just predict the next token faster or with more parameters. They allocate variable compute at inference time — spending more steps on harder problems and fewer on easy ones. The result: dramatic improvements on math, coding, science, and multi-step reasoning tasks that previous models plateaued on.

This post covers three questions:
1. **How do thinking models work?** — the inference-time mechanism
2. **How do you build one?** — the training pipeline from SFT to RL
3. **What data do you need?** — the types, sources, and design of training data

---

## 1. What Is a Thinking Model?

A thinking model is an LLM that generates an explicit **chain of internal reasoning** (often called a "thinking trace" or "extended thinking") before producing its final answer.

```
User: How many r's are in "strawberry"?

<thinking>
Let me spell out "strawberry" letter by letter:
s-t-r-a-w-b-e-r-r-y
Now I'll identify each 'r':
- Position 3: r
- Position 8: r
- Position 9: r
That's 3 r's total.
</thinking>

Answer: There are 3 r's in "strawberry."
```

The thinking trace is **not just a prompt engineering trick**. It's a fundamentally different inference paradigm: the model is trained to decompose problems, try sub-strategies, backtrack when stuck, and self-verify — all within a single forward generation.

### Where Does Reasoning Come From?

A common misconception is that reasoning is purely a post-training phenomenon. In reality, reasoning capability is developed across **all three stages** of the model lifecycle:

| Stage | What Happens | Role in Reasoning |
|-------|-------------|-------------------|
| **Pretraining** | Model learns from trillions of tokens including textbooks, proofs, code, scientific papers | Builds latent reasoning circuits — the model learns logical patterns, mathematical operations, and step-by-step derivation styles from data that *already contains reasoning* |
| **Post-Training (SFT + RL)** | Supervised fine-tuning on reasoning traces + RL with outcome rewards | **Shapes and amplifies** reasoning into a structured, reliable behavior. Teaches the `<thinking>` format, backtracking, self-verification |
| **Inference Time** | Chain-of-thought prompting, best-of-N sampling, tree search | Elicits and scales reasoning at serving time — even base models can reason with proper prompting |

**Pretraining lays the foundation.** A base model pretrained on high-quality data (math textbooks, code repositories, scientific literature) already has latent reasoning capabilities. This is why chain-of-thought prompting works on base models at all — the reasoning patterns are *already learned*, just not reliably activated. Models pretrained on more reasoning-rich corpora (code, formal proofs, step-by-step solutions) show stronger reasoning baselines before any post-training.

**Post-training shapes and amplifies.** What SFT + RL adds is not reasoning *ability* from scratch, but rather: (1) a structured format (`<thinking>` blocks) to reliably surface reasoning, (2) self-correction and backtracking behaviors trained via RL reward signals, and (3) adaptive depth — knowing when to think deeply vs. briefly.

**Inference-time techniques unlock more.** Even after training, prompting strategies (few-shot CoT, self-consistency via majority voting) and search methods (best-of-N, MCTS with process reward models) further improve reasoning by exploring more of the solution space at serve time.

> **The practical implication:** If you're building a thinking model, don't ignore pretraining data composition. Including more mathematical derivations, annotated code, and structured problem-solving in the pretraining mix gives post-training a much stronger foundation to build on. Teams that treat reasoning as *only* a post-training problem leave significant capability on the table.

### Standard LLM vs. Thinking Model

| Aspect | Standard LLM | Thinking Model |
|--------|-------------|----------------|
| **Inference** | Single-pass token generation | Extended reasoning → then answer |
| **Compute allocation** | Fixed per token | Variable — more thinking for harder problems |
| **Error correction** | Limited (autoregressive, can't "go back") | Can backtrack, try alternatives within the thinking trace |
| **Training signal** | Next-token prediction + RLHF | Reasoning-rich pretraining + SFT on traces + RL with outcome rewards |
| **Latency** | Lower | Higher (more tokens generated) |
| **Strengths** | Fluency, knowledge recall, simple tasks | Multi-step reasoning, math, coding, planning |

### The Core Insight: Test-Time Compute Scaling

The key idea from the "Scaling LLM Test-Time Compute" line of research (Snell et al., 2024) is that **you can trade inference-time compute for capability**. Instead of making the model bigger or training longer, let the model "think more" on hard problems.

This creates a new scaling axis: while pretraining scaling laws relate capability to parameters and data, **test-time scaling laws** relate capability to the number of reasoning tokens generated. The empirical finding: on reasoning-heavy tasks, doubling thinking tokens can yield improvements comparable to a 10× increase in model parameters.

$$
\text{Performance} \propto \log(\text{test-time compute budget})
$$

This is why thinking models are transformative — they unlock a way to improve performance without retraining.

---

## 2. How Thinking Models Work (Inference)

At inference time, a thinking model's generation has two phases:

### Phase 1: Extended Thinking

The model generates a reasoning trace, typically wrapped in special tokens like `<thinking>...</thinking>`. Within this trace, the model:

- **Decomposes** the problem into sub-problems
- **Explores** multiple solution paths
- **Backtracks** when it hits dead ends ("Wait, that approach doesn't work because...")
- **Verifies** intermediate results ("Let me double-check: 7 × 8 = 56, yes")
- **Summarizes** findings before moving to the answer

This is not hardcoded behavior — it's **emergent from training**. The model learned that these patterns lead to higher rewards on verification tasks.

### Phase 2: Final Answer

After the thinking block, the model produces a concise answer. The thinking trace is typically hidden from the user (or shown optionally), so the user experience is: ask a question → get a well-reasoned answer.

### What Happens Inside the Thinking Trace

A real thinking trace on a math competition problem might look like this (simplified):

```
<thinking>
I need to find all positive integers n such that n² + 2n + 3 is a perfect square.

Let me set n² + 2n + 3 = k² for some positive integer k.

Rearranging: k² - n² = 2n + 3
(k-n)(k+n) = 2n + 3

Since n² + 2n + 3 = (n+1)² + 2, we need (n+1)² + 2 = k².
So k² - (n+1)² = 2
(k - n - 1)(k + n + 1) = 2

Since k > n+1 (because k² = (n+1)² + 2 > (n+1)²), both factors are positive.
The only way to factor 2 as a product of two positive integers is 1 × 2.

So: k - n - 1 = 1 and k + n + 1 = 2
From the second equation: k = 1 - n
But then k - n - 1 = 1 - n - n - 1 = -2n, which must equal 1.
So -2n = 1, giving n = -1/2. Not a positive integer.

Wait — let me reconsider. Maybe I should check small values directly.
n=1: 1+2+3=6, not a perfect square.
n=2: 4+4+3=11, no.
n=3: 9+6+3=18, no.
...

Actually, my algebra showed there's no solution. The factoring argument
proves k² - (n+1)² = 2 has no positive integer solutions.

So the answer is: there are no positive integers n with this property.
</thinking>
```

Notice the key cognitive patterns:
1. **Setting up equations** (structured approach)
2. **Algebraic manipulation** (trying a path)
3. **Getting stuck and backtracking** ("Wait — let me reconsider")
4. **Switching strategies** (direct computation as a check)
5. **Synthesizing** (combining both approaches for the final answer)

These patterns aren't scripted — they emerge from RL training on problems with verifiable answers.

---

## 3. How to Build a Thinking Model

The training recipe follows a multi-stage pipeline. Based on public research from DeepSeek-R1 (Guo et al., 2025), OpenAI's o1 blog posts, and QwQ/Qwen, the consensus approach looks like this:

```
Pretrained Base Model
       │
       ▼
Stage 1: Cold-Start SFT (seed reasoning behavior)
       │
       ▼
Stage 2: RL with Outcome-Based Rewards (scale reasoning)
       │
       ▼
Stage 3: Rejection Sampling + SFT (distill & polish)
       │
       ▼
Stage 4: RL Alignment (human preferences + safety)
       │
       ▼
Thinking Model
```

### Stage 1: Cold-Start SFT — Seeding the Reasoning Format

**Goal:** Teach the model to *produce* reasoning traces in the right format.

A pretrained base model doesn't naturally generate `<thinking>` blocks with step-by-step reasoning. The cold-start SFT stage fine-tunes on a small set of high-quality reasoning demonstrations to bootstrap this behavior.

**What the training data looks like:**

```json
{
  "prompt": "Solve: What is the sum of all prime numbers less than 20?",
  "response": "<thinking>\nLet me list all prime numbers less than 20.\nA prime number is only divisible by 1 and itself.\n\n2 - prime\n3 - prime\n4 = 2×2, not prime\n5 - prime\n6 = 2×3, not prime\n7 - prime\n8 = 2×4, not prime\n9 = 3×3, not prime\n10 = 2×5, not prime\n11 - prime\n12 = 2×6, not prime\n13 - prime\n14 = 2×7, not prime\n15 = 3×5, not prime\n16 = 2×8, not prime\n17 - prime\n18 = 2×9, not prime\n19 - prime\n\nPrimes less than 20: 2, 3, 5, 7, 11, 13, 17, 19\nSum = 2+3+5+7+11+13+17+19 = 77\n</thinking>\n\nThe sum of all prime numbers less than 20 is **77**."
}
```

**Key design choices:**

| Decision | Common Approach | Why |
|----------|----------------|-----|
| Data size | 1K–10K examples | Small but high quality; just enough to establish format |
| Source | Expert-written or distilled from a stronger model | Quality over quantity at this stage |
| Format tokens | `<thinking>`, `<answer>` or similar delimiters | Gives the model a clear structure to generate |
| Reasoning style | Verbose, step-by-step, with self-correction | Establish the *type* of reasoning the model should produce |

> **DeepSeek-R1's approach:** They collected thousands of CoT examples using few-shot prompting with a long-CoT format. These included reflection and verification behaviors — patterns like "Let me verify," "Wait, that's wrong," and "I'll try another approach." The cold-start data explicitly models these cognitive patterns.

### Stage 2: RL with Outcome-Based Rewards — Where Thinking Emerges

**Goal:** Scale and improve reasoning quality via reinforcement learning.

This is the critical stage. After cold-start SFT gives the model the *format*, RL gives it the *substance*. The model learns that better reasoning → better answers → higher rewards.

**The RL setup:**

```python
# Simplified GRPO training loop for thinking models
for batch in training_data:
    prompts = batch["prompts"]  # math/code/reasoning problems
    
    # Sample G completions per prompt (with thinking traces)
    completions = []
    for prompt in prompts:
        group = [model.generate(prompt, temperature=0.7) for _ in range(G)]
        completions.append(group)
    
    # Outcome-based reward: did the final answer match?
    rewards = []
    for prompt, group in zip(prompts, completions):
        group_rewards = []
        for completion in group:
            answer = extract_answer(completion)  # parse answer after </thinking>
            reward = verify(answer, ground_truth[prompt])  # 1.0 if correct, 0.0 otherwise
            reward += format_reward(completion)  # small bonus for proper formatting
            group_rewards.append(reward)
        rewards.append(group_rewards)
    
    # GRPO update: within-group advantage normalization
    update_policy(model, completions, rewards)
```

**Why outcome-based rewards work so well here:**

The beauty of this approach is that **the reward function doesn't specify *how* to think** — only whether the final answer is correct. The model discovers effective reasoning strategies on its own. This is why behaviors like backtracking, self-verification, and exploring multiple paths emerge spontaneously — they're selected for because they lead to correct answers.

| Reward Component | Signal | Purpose |
|-----------------|--------|---------|
| **Correctness reward** | 1.0 if answer matches ground truth, 0.0 otherwise | Primary driver of reasoning quality |
| **Format reward** | Small bonus for proper `<thinking>...</thinking>` structure | Maintain structured output |
| **Length penalty** (optional) | Penalize excessively long thinking traces | Prevent reward hacking via verbosity |

> **What DeepSeek-R1 observed:** During RL training, the model spontaneously learned to allocate more thinking tokens to harder problems. On easy arithmetic, the thinking trace might be 50 tokens; on AIME problems, it could exceed 10,000 tokens. This **adaptive compute allocation** was not explicitly trained — it emerged from the reward signal.

**The "aha moment":** DeepSeek-R1's paper describes a fascinating training dynamic — at some point during RL training, the model begins to re-evaluate its initial approach and try alternative strategies within a single thinking trace. This emergent self-correction behavior is the hallmark of a thinking model and appears to be a phase transition in training.

### Stage 3: Rejection Sampling + SFT — Distillation and Polish

**Goal:** Stabilize and improve the quality of reasoning traces.

After RL, the model produces correct answers more often, but its reasoning traces can be messy — repetitive, overly verbose, or containing unnecessary detours. This stage cleans things up:

1. **Sample many responses** per prompt from the RL-trained model
2. **Filter for correct answers** (rejection sampling)
3. **Select the best reasoning trace** among correct answers (shortest, clearest, most well-structured)
4. **Fine-tune on these curated traces** (SFT)

```python
# Rejection sampling pipeline
curated_data = []
for prompt in training_prompts:
    candidates = [model.generate(prompt) for _ in range(N)]  # N = 16-64
    correct = [c for c in candidates if verify(extract_answer(c))]
    
    if correct:
        # Select best trace: balance quality, clarity, and length
        best = select_best_trace(correct, criteria=[
            "correctness",
            "reasoning_clarity", 
            "minimal_redundancy",
            "appropriate_length"
        ])
        curated_data.append({"prompt": prompt, "response": best})
```

This creates a high-quality SFT dataset *from the model's own successful reasoning*, which is then used for another round of fine-tuning. The result: the model maintains the reasoning capability from RL but with cleaner, more consistent output.

### Stage 4: RL Alignment — Human Preferences and Safety

**Goal:** Align the thinking model with human preferences while preserving reasoning capability.

A pure outcome-optimized model might be brilliant at math but terrible at conversation — it might refuse to engage with open-ended questions, produce unhelpful thinking traces, or ignore user instructions.

This final stage applies standard alignment techniques (RLHF/DPO) with some thinking-model-specific considerations:

- **Helpfulness on non-reasoning tasks:** The model should not "overthink" simple questions
- **Thinking trace quality:** Even though users may not see the trace, its quality affects the final answer
- **Safety in reasoning:** The model should not reason its way into harmful outputs
- **Format consistency:** Decide when to show thinking vs. directly answering

---

## 4. Training Data for Thinking Models

Data is the most underappreciated component of building thinking models. The types and quality of training data at each stage dramatically affect the final model's reasoning capabilities.

The four subsections below **mirror the training pipeline order**: cold-start SFT data (4.1) is used first, then RL training data (4.2), then rejection sampling data (4.3) is generated from the RL-trained model, and finally alignment data (4.4) is used in the last stage. Within each subsection, however, the listed data sources are **alternatives or complements** — you mix and match them, not apply them sequentially.

### 4.1 Cold-Start SFT Data: Reasoning Demonstrations

**Purpose:** Teach the model the format and style of extended reasoning.

| Data Source | Description | Strength | Weakness |
|-------------|------------|----------|----------|
| **Human-written CoT** | Experts write detailed step-by-step solutions | Highest quality reasoning patterns | Expensive, doesn't scale |
| **Distillation from stronger models** | Prompt GPT-4 / Claude to "think step by step" and capture traces | Scalable, high quality | Ceiling bounded by teacher model |
| **Few-shot elicitation** | Use few-shot examples to elicit CoT from the base model itself | Self-consistent style | Lower quality than external distillation |
| **Existing datasets with solutions** | Math competition solutions, coding editorial, scientific derivations | Naturally high quality, diverse | Requires reformatting into thinking-trace format |

**Key quality criteria for cold-start data:**

1. **Explicit reasoning steps** — not just answers, but the full derivation
2. **Self-correction patterns** — include examples where the reasoning backtracks
3. **Verification steps** — "Let me check: ..." patterns
4. **Variable difficulty** — from simple arithmetic to competition-level problems
5. **Diverse domains** — math, code, logic, science, common-sense reasoning

> **Practical tip from DeepSeek-R1:** Their cold-start data included explicit markers for cognitive transitions: "Let me think about this differently," "Wait, I made an error," "To verify, I'll..." These markers helped the model learn *when* to apply these strategies, not just that they exist.

### 4.2 RL Training Data: Problems with Verifiable Answers

**Purpose:** Provide the training signal (correct/incorrect) that drives reasoning improvement through RL.

This is the most critical data requirement: **you need large quantities of problems where correctness can be automatically verified.** Human judgment doesn't scale for RL — you need millions of reward signal evaluations.

| Domain | Data Source | Verification Method | Scale Available |
|--------|-----------|-------------------|----------------|
| **Math** | GSM8K, MATH, AIME, competition archives, synthetic problems | Exact numerical match | 100K–1M+ (with augmentation) |
| **Code** | HumanEval, MBPP, LeetCode, CodeContests, SWE-bench | Unit test execution | 50K–500K |
| **Logic puzzles** | Formal logic, constraint satisfaction, game solving | Formal verification / rule checking | 10K–100K |
| **Science** | Physics / chemistry problems with numerical answers | Exact match / tolerance check | 10K–50K |
| **Formal math** | Lean, Isabelle, Coq theorem proving | Proof checker | 10K–100K |

**Scaling RL training data with synthetic generation:**

The bottleneck is usually not model capability but *problem supply*. Techniques to scale:

```python
# Example: Synthetic math problem generation
def generate_math_problems(template, difficulty_range, n_problems):
    """Generate variations of math problems with known solutions."""
    problems = []
    for _ in range(n_problems):
        # Vary parameters while keeping problem structure
        params = sample_parameters(template, difficulty_range)
        problem_text = template.format(**params)
        solution = compute_solution(template, params)  # symbolic or numerical
        problems.append({
            "prompt": problem_text,
            "ground_truth": solution,
            "difficulty": estimate_difficulty(params)
        })
    return problems
```

**The difficulty curriculum matters:**

| Training Phase | Problem Difficulty | Rationale |
|---------------|-------------------|-----------|
| Early RL | Easy–medium (>50% solve rate) | Model needs positive reward signal to learn |
| Mid RL | Medium–hard (20–50% solve rate) | Push the frontier of reasoning capability |
| Late RL | Hard (5–20% solve rate) | Develop sophisticated multi-step strategies |

If problems are too hard too early, the model gets near-zero reward and learns nothing. If problems are too easy throughout, the model never develops deep reasoning. **Curriculum design is as important as the RL algorithm.**

### 4.3 Rejection Sampling Data: Self-Generated Traces

**Purpose:** Create high-quality SFT data from the model's own successful reasoning.

This data comes from the model itself — you don't need external annotation. The pipeline:

1. Take a large set of problems (can overlap with RL training data or be new)
2. Sample many completions per problem (N=16–64) from the RL-trained model
3. Filter for correct final answers
4. Among correct answers, select the best reasoning trace
5. The resulting (prompt, best_trace) pairs become SFT training data

**What makes a "best" trace among correct ones?**

| Criterion | How to Measure | Why It Matters |
|-----------|---------------|----------------|
| **Conciseness** | Token count | Shorter correct traces are more efficient |
| **Logical flow** | LLM judge or heuristics | Clear reasoning → better downstream SFT signal |
| **Minimal backtracking** | Count "wait"/"actually" patterns | Some backtracking is good; excessive is wasteful |
| **Verification presence** | Regex/LLM check for "let me verify" | Self-verification → more reliable reasoning |

### 4.4 Alignment Data: Preference Pairs for Reasoning

**Purpose:** Human preference data to align the thinking model with user expectations.

This data is similar to standard RLHF preference data, but with reasoning-specific considerations:

**What labelers evaluate:**

- Final answer correctness (most important)
- Reasoning trace clarity (when shown to user)
- Appropriate depth — not overthinking trivial questions
- Following user instructions about format and style
- Safety and refusal behavior in reasoning

**Reasoning-specific preference patterns:**

```
Prompt: "What's 2 + 2?"

Response A (rejected — overthinking):
<thinking>
Let me carefully consider this arithmetic problem. 
2 + 2 means I need to add the integer 2 to itself.
In the natural numbers, addition is defined recursively...
[500 tokens of unnecessary reasoning]
</thinking>
The answer is 4.

Response B (preferred — appropriate depth):
4.
```

```
Prompt: "Prove that √2 is irrational."

Response A (rejected — too shallow):
√2 is irrational because it can't be expressed as a fraction.

Response B (preferred — appropriate depth):
<thinking>
I'll use proof by contradiction.
Assume √2 = p/q where p, q are integers with no common factors.
Then 2 = p²/q², so p² = 2q².
This means p² is even, so p must be even. Let p = 2k.
Then (2k)² = 2q², so 4k² = 2q², so q² = 2k².
This means q is also even — contradicting our assumption 
that p and q have no common factors.
Therefore √2 is irrational.
</thinking>

**Proof by contradiction:** Assume √2 = p/q in lowest terms...
```

### 4.5 Data Taxonomy Summary

| Stage | Data Type | Volume | Key Property | Source |
|-------|-----------|--------|-------------|--------|
| Cold-Start SFT | Reasoning demonstrations | 1K–10K | High quality, explicit reasoning patterns | Expert-written, distilled, existing solutions |
| RL Training | Problems with verifiable answers | 100K–1M+ | Auto-verifiable correctness | Math datasets, code problems, synthetic generation |
| Rejection Sampling | Self-generated successful traces | 100K–500K | Best-of-N from model's own output | Automated pipeline (no human annotation) |
| Alignment | Preference pairs with reasoning | 10K–100K | Human judgment on answer + trace quality | Human annotators |

---

## 5. Key Design Decisions and Open Questions

### To Show or Hide the Thinking Trace?

| Option | Pros | Cons |
|--------|------|------|
| **Always hidden** (o1) | Cleaner UX, protects IP | Users can't verify reasoning, harder to debug |
| **Always shown** (DeepSeek-R1) | Transparent, educational, debuggable | Verbose, may confuse non-technical users |
| **User-controlled toggle** (Claude 3.7) | Best of both worlds | More complex UX, users must understand the option |

### Budget Forcing: Controlling Thinking Length

A practical concern: thinking models can "run away" with excessively long reasoning traces, burning compute without improving answer quality.

**Budget forcing** constrains the thinking trace to a specified token budget. Techniques include:

- Hard token limit on the `<thinking>` block
- End-of-thinking tokens injected by the serving system
- RL training with length penalties to learn natural stopping points
- User-specified "thinking effort" levels (e.g., Claude's "low/medium/high" extended thinking)

### When Not to Think

Not every query benefits from extended reasoning. An ideal thinking model should learn to:

- **Think deeply** on: math, code, logic, planning, analysis
- **Think briefly** on: factual recall, translation, formatting
- **Skip thinking** on: greetings, simple Q&A, creative writing (arguably)

This routing behavior can be trained through the alignment stage by including preference data that penalizes unnecessary thinking.

### Distillation: Small Thinking Models

A powerful strategy: train a large thinking model (70B+), then distill its reasoning traces into a smaller model (7B–14B) via SFT. DeepSeek-R1 demonstrated that distilled 14B and 32B models can significantly outperform non-thinking models of the same size — and even outperform larger non-thinking models.

The distillation pipeline:

```
Large Thinking Model (70B)
  → Generate reasoning traces on diverse problems
  → Filter for correct + high-quality traces
  → SFT smaller model on these traces
Small Thinking Model (7B-14B)
```

> **Surprising finding from DeepSeek-R1:** Distillation outperformed applying RL directly to smaller models for reasoning tasks. The reasoning patterns generated by the larger model provided a stronger learning signal than what smaller models could discover via RL on their own. This suggests that **reasoning capability transfers well through distillation**, even across significant parameter count differences.

---

## 6. The Broader Picture

### Why Thinking Models Matter

1. **New scaling axis:** When model size and data scaling plateau, test-time compute scaling offers a fresh dimension of improvement
2. **Verifiable reasoning:** Unlike standard LLMs where you trust the output, thinking traces provide an **auditable reasoning chain**
3. **Adaptive compute:** Pay for thinking only when needed — simple tasks stay fast
4. **Foundation for agents:** Extended reasoning is essential for planning, tool use, and multi-step task execution

### What's Next

- **Multi-modal thinking:** Reasoning over images, diagrams, and mixed media
- **Thinking with tools:** Interleaving reasoning with code execution, search, and API calls — not just text-based thinking
- **Efficient thinking:** Reducing the compute cost of extended reasoning through parallel verification, speculative thinking, and learned compression
- **Thinking about thinking:** Meta-cognitive models that can reason about their own reasoning process and allocate thinking budget optimally

---

## References

### Thinking Models
1. **OpenAI o1** — "Learning to Reason with LLMs" (2024) — First major thinking model release
2. **DeepSeek-R1** — Guo et al., 2025 — Open-source thinking model with detailed training methodology
3. **QwQ** — Qwen Team, 2024 — Open-weight thinking model

### Test-Time Compute Scaling
4. **Scaling LLM Test-Time Compute Optimally** — Snell et al., 2024 — Foundational work on inference-time scaling laws
5. **Let's Verify Step by Step** — Lightman et al., 2023 — Process reward models for math reasoning

### RL for Reasoning
6. **DeepSeekMath** — Shao et al., 2024 — GRPO for mathematical reasoning
7. **Star** — Zelikman et al., 2022 — Self-Taught Reasoner: bootstrapping reasoning via rationalization
8. **ReST** — Gulcehre et al., 2023 — Reinforced Self-Training for language models
9. **STILL-2** — Slow Thinking with LLMs, 2024 — Training recipe analysis for thinking models

### Chain-of-Thought and Reasoning
10. **Chain-of-Thought Prompting** — Wei et al., 2022 — The foundational prompting technique
11. **Self-Consistency** — Wang et al., 2023 — Sampling multiple reasoning paths and majority voting
12. **Tree of Thoughts** — Yao et al., 2023 — Deliberate problem solving with LLMs

### Post-Training Foundations
13. **PPO** — Schulman et al., 2017 — Proximal Policy Optimization
14. **GRPO** — Group Relative Policy Optimization (used in DeepSeek-R1 and DeepSeekMath)
15. **DPO** — Rafailov et al., 2023 — Direct Preference Optimization
