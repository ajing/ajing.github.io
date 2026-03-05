---
author: Jing Lu
pubDatetime: 2026-03-05T00:00:00Z
title: "Instruction Following: What Models Get Wrong and How to Fix It with Better Post-Training Data"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Post-Training
description: "LLMs can write poetry and solve math, but ask them to 'respond in exactly 3 bullet points using only lowercase' and they stumble. This post dissects the taxonomy of instruction-following failures and provides a practical playbook for building post-training data that actually fixes them."
---

**LLMs are impressively capable, yet frustratingly unreliable** at following precise instructions. A model that can write a Shakespearean sonnet may fail at "respond in exactly 3 sentences" or "format your answer as JSON." This gap between general intelligence and instruction adherence is one of the most practically important problems in post-training — and one where targeted data preparation makes a measurable difference.

This post covers three things: (1) a taxonomy of instruction-following questions that models face, (2) why models fail at them, and (3) a concrete playbook for preparing post-training data that systematically improves instruction adherence.

---

## 1. What Is Instruction Following, Really?

Instruction following is the model's ability to satisfy **explicit constraints** in a user prompt — constraints on format, length, content, style, or structure — *in addition to* answering the underlying question correctly.

It's distinct from:
- **Task completion** (can the model solve the problem?) — a model can solve a math problem correctly but output it as prose when JSON was requested
- **Safety alignment** (does the model refuse harmful requests?) — orthogonal concern
- **Helpfulness** (is the response useful?) — a helpful response that ignores format constraints is still a failure

The key insight: **instruction following is a constraint satisfaction problem layered on top of generation**. The model must simultaneously produce high-quality content AND satisfy a set of hard constraints. This dual objective is what makes it hard.

---

## 2. Taxonomy of Instruction-Following Constraints

Based on the IFEval benchmark, production failure analysis, and real-world API usage patterns, instruction-following constraints fall into roughly seven categories:

### 2.1 Format Constraints

The most common and most measurably violated category:

| Constraint Type | Example | Failure Mode |
|----------------|---------|--------------|
| **Output structure** | "Respond in JSON format" | Returns prose, missing fields, trailing commas, wrong types |
| **Markup/annotation** | "Use markdown headers for each section" | Inconsistent header levels, missing headers |
| **Delimiters** | "Separate items with `\|`" | Uses commas or newlines instead |
| **Enclosure** | "Wrap each keyword in double quotes" | Single quotes, no quotes, or inconsistent quoting |
| **Casing** | "Title Case all headers" / "ALL UPPERCASE" | Reverts to natural casing mid-response |

**Why it matters:** Format constraints are the gateway to reliable tool integration. An API that returns malformed JSON crashes downstream systems. A report generator that ignores markdown structure produces unreadable documents.

### 2.2 Length Constraints

| Constraint Type | Example | Failure Mode |
|----------------|---------|--------------|
| **Word count** | "Respond in exactly 50 words" | Off by 20-40%, especially undershooting |
| **Sentence count** | "Use exactly 3 sentences" | 4-5 sentences, or run-on sentences |
| **Paragraph count** | "Write 2 paragraphs" | 3-4 paragraphs, or one long block |
| **Min/max bounds** | "At least 200 words but no more than 300" | Under minimum or ignores maximum |
| **Character limits** | "Your response must be under 280 characters" | Exceeds by 50-100 characters |

**Interesting finding:** Models are significantly better at *minimum* length constraints than *maximum* length constraints. The generative bias toward verbosity makes it easier to exceed a floor than respect a ceiling.

### 2.3 Content Inclusion/Exclusion

| Constraint Type | Example | Failure Mode |
|----------------|---------|--------------|
| **Must include** | "Include the phrase 'in conclusion'" | Omits the required phrase |
| **Must exclude** | "Do not mention competitors" | Mentions competitors in passing |
| **Keyword placement** | "Start your response with 'Certainly'" | Starts with a different opener |
| **Topic restriction** | "Only discuss benefits, not drawbacks" | Includes caveats or downsides |
| **Specific vocabulary** | "Use only technical terminology" | Mixes casual and technical language |

### 2.4 Structural/Organizational Constraints

| Constraint Type | Example | Failure Mode |
|----------------|---------|--------------|
| **List format** | "Answer as a numbered list" | Uses bullets or prose |
| **Section structure** | "Organize into Pros, Cons, and Verdict" | Missing sections or different names |
| **Ordering** | "List items from most to least important" | Random or alphabetical ordering |
| **Table format** | "Present as a markdown table with columns X, Y, Z" | Missing columns, wrong alignment |

### 2.5 Stylistic/Tone Constraints

| Constraint Type | Example | Failure Mode |
|----------------|---------|--------------|
| **Reading level** | "Explain like I'm 5" | Uses jargon or complex sentence structure |
| **Persona** | "Respond as a pirate" | Drops persona after first sentence |
| **Formality** | "Use formal academic language" | Slips into casual tone |
| **Perspective** | "Write in second person" | Switches between first and second person |

### 2.6 Language/Locale Constraints

| Constraint Type | Example | Failure Mode |
|----------------|---------|--------------|
| **Response language** | "Respond in French" | Mixes in English technical terms |
| **Script** | "Use only Cyrillic characters" | Includes Latin characters for proper nouns |
| **Date format** | "Use DD/MM/YYYY format" | Defaults to MM/DD/YYYY |
| **Number format** | "Use European number formatting (1.000,50)" | Uses US formatting (1,000.50) |

### 2.7 Multi-Constraint Compositions

The hardest category — combining two or more constraints:

> "Respond in exactly 3 bullet points, each under 20 words, in formal French, using only present tense."

**This is where models break down most dramatically.** Each individual constraint might be satisfied 85-90% of the time, but the *joint* satisfaction rate drops multiplicatively. Four independent constraints at 85% each yields ~52% joint success. In practice, it's worse because constraints interact — maintaining exact word count while switching to a formal register is harder than either alone.

---

## 3. Why Models Fail at Instruction Following

Understanding failure modes is essential for designing data that fixes them:

### 3.1 The Autoregressive Trap

LLMs generate left-to-right. When a constraint requires **global properties** of the response (e.g., "exactly 50 words"), the model must plan ahead — but it doesn't have a reliable mechanism for this. It can estimate word count as it generates, but errors accumulate:

```
Token 1-10:  "I have about 40 words left" (reasonable estimate)
Token 30-40: "I have about 15 words left" (drift from actual count)
Token 45-55: Overshoots or undershoots by 10-20%
```

**Implication for data:** Training examples should include responses where the model *demonstrably counted correctly*. The model needs to learn internal counting heuristics from examples.

### 3.2 Instruction Dilution in Long Contexts

As the model generates more tokens, the attention weight on the original instruction decreases. This manifests as:
- **Persona drift:** Role-play instructions followed for the first paragraph, then abandoned
- **Format drift:** JSON structure maintained for the first few fields, then malformed
- **Style drift:** Formal tone for the first sentence, then casual

**Implication for data:** Training examples should be long enough to test persistence. A 50-token response that follows a format constraint teaches less than a 500-token response that maintains it throughout.

### 3.3 Conflicting Pretraining Priors

The model's pretraining distribution creates strong priors that instruction following must override:
- Trained on mostly English text → resists generating entirely in another language
- Trained on natural prose → resists structured output (JSON, tables)
- Trained on helpful, verbose responses → resists concise formats
- Trained on balanced, nuanced text → resists one-sided arguments when instructed

**Implication for data:** The harder the constraint fights the prior, the more training examples you need. "Write in JSON" requires more coverage than "use bullet points" because prose is the stronger prior.

### 3.4 Reasoning vs. Constraint Satisfaction Trade-off

A surprising finding from Amazon Science (2024): **explicit chain-of-thought reasoning can degrade instruction-following accuracy**. The reasoning tokens consume attention capacity, diverting the model from instruction-relevant tokens. The model gets so focused on "thinking through the problem" that it forgets the formatting constraint.

**Implication for data:** When building data for reasoning tasks with format constraints, include examples where the model reasons *and* follows constraints — not examples where reasoning quality is sacrificed for format compliance.

---

## 4. Preparing Post-Training Data for Better Instruction Following

This is the core section — a practical playbook for building SFT and RL data that systematically improves instruction adherence.

### 4.1 The Data Taxonomy: Constraint-Labeled Training Examples

The first step is to **tag every training example with the constraints it exercises**:

```python
@dataclass
class InstructionExample:
    prompt: str
    response: str
    constraints: List[Constraint]
    constraint_satisfaction: Dict[str, bool]  # Verified per-constraint

@dataclass
class Constraint:
    category: str      # "format", "length", "content", "style", etc.
    subcategory: str   # "json_output", "word_count", "persona", etc.
    specification: str # "exactly 50 words", "respond as JSON", etc.
    verifiable: bool   # Can this be checked algorithmically?
    verifier: Optional[Callable]  # Function to check satisfaction
```

**Why label constraints?** Without labels, you can't measure coverage. Your training set might have 10,000 examples but only 50 that test JSON output. Constraint labeling exposes these gaps.

### 4.2 Building Verifiable Constraint Checkers

The superpower of instruction-following data is that **many constraints are algorithmically verifiable**. Unlike general helpfulness (which requires human judgment), you can write code to check:

```python
VERIFIERS = {
    "word_count_exact": lambda resp, n: len(resp.split()) == n,
    "word_count_min": lambda resp, n: len(resp.split()) >= n,
    "word_count_max": lambda resp, n: len(resp.split()) <= n,
    "sentence_count": lambda resp, n: len(sent_tokenize(resp)) == n,
    "starts_with": lambda resp, prefix: resp.strip().startswith(prefix),
    "ends_with": lambda resp, suffix: resp.strip().endswith(suffix),
    "contains_phrase": lambda resp, phrase: phrase.lower() in resp.lower(),
    "excludes_phrase": lambda resp, phrase: phrase.lower() not in resp.lower(),
    "all_uppercase": lambda resp, _: resp == resp.upper(),
    "all_lowercase": lambda resp, _: resp == resp.lower(),
    "valid_json": lambda resp, _: is_valid_json(resp),
    "markdown_headers": lambda resp, n: resp.count("\n#") >= n,
    "bullet_count": lambda resp, n: len(re.findall(r"^[-*•]\s", resp, re.M)) == n,
    "no_questions": lambda resp, _: "?" not in resp,
    "language_check": lambda resp, lang: detect_language(resp) == lang,
}

def verify_all_constraints(response: str, constraints: List[Constraint]) -> Dict[str, bool]:
    results = {}
    for c in constraints:
        if c.verifiable and c.subcategory in VERIFIERS:
            results[c.subcategory] = VERIFIERS[c.subcategory](response, c.specification)
        else:
            results[c.subcategory] = None  # Requires human judgment
    return results
```

**This is the foundation of your data pipeline.** Every generated response can be automatically checked against its constraints. Bad examples are filtered out. Good examples are kept. The verifiers become your quality gate.

### 4.3 Synthetic Data Generation Pipeline

The most scalable approach for instruction-following data is **synthesis with verification**:

#### Step 1: Generate Diverse Constraint Sets

```python
def generate_constraint_set(
    categories: List[str],
    n_constraints: int = 2,
    difficulty: str = "medium"
) -> List[Constraint]:
    """
    Generate a random combination of constraints.
    
    Difficulty levels:
    - easy: 1 constraint, common types (bullet list, word count)
    - medium: 2-3 constraints, includes format + length
    - hard: 3-5 constraints, includes conflicting priors
    """
    pool = CONSTRAINT_TEMPLATES[difficulty]
    selected = random.sample(pool, min(n_constraints, len(pool)))
    
    # Check for impossible combinations
    if is_contradictory(selected):
        return generate_constraint_set(categories, n_constraints, difficulty)
    
    return selected
```

Constraint templates by difficulty:

| Difficulty | Example Constraint Set | Joint Satisfaction Target |
|-----------|----------------------|--------------------------|
| **Easy** | "Use bullet points" | 95%+ |
| **Medium** | "Respond in JSON" + "under 100 words" | 85%+ |
| **Hard** | "3 paragraphs" + "formal French" + "no questions" + "include the word 'innovation'" | 70%+ |
| **Adversarial** | "Exactly 50 words" + "ALL CAPS" + "respond as a haiku-writing pirate" | 50%+ |

#### Step 2: Generate Seed Tasks

Combine constraint sets with diverse task types:

```python
TASK_TEMPLATES = [
    "Explain {topic} to a {audience}.",
    "Compare {thing_a} and {thing_b}.",
    "Write a {document_type} about {subject}.",
    "Summarize the following text: {text}",
    "List the pros and cons of {decision}.",
    "Draft an email about {situation}.",
    "Create a {content_type} for {purpose}.",
]

def generate_prompt(task_template: str, constraints: List[Constraint]) -> str:
    task = fill_template(task_template)
    constraint_text = "\n".join([
        f"- {c.specification}" for c in constraints
    ])
    return f"{task}\n\nRequirements:\n{constraint_text}"
```

#### Step 3: Generate and Verify Responses

Use a strong teacher model (or the model being trained) with rejection sampling:

```python
async def generate_verified_examples(
    prompts: List[str],
    constraints: List[List[Constraint]],
    teacher_model: str,
    n_samples: int = 8,
    temperature: float = 0.7
) -> List[InstructionExample]:
    verified_examples = []
    
    for prompt, constraint_set in zip(prompts, constraints):
        # Sample multiple responses
        responses = await asyncio.gather(*[
            generate(teacher_model, prompt, temperature=temperature)
            for _ in range(n_samples)
        ])
        
        # Verify each response against all constraints
        for response in responses:
            satisfaction = verify_all_constraints(response, constraint_set)
            
            # Keep only fully constraint-satisfying responses
            if all(v is True for v in satisfaction.values() if v is not None):
                verified_examples.append(InstructionExample(
                    prompt=prompt,
                    response=response,
                    constraints=constraint_set,
                    constraint_satisfaction=satisfaction,
                ))
                break  # One verified example per prompt is sufficient
    
    return verified_examples
```

**Critical insight: rejection sampling is your best friend.** A teacher model might satisfy all constraints only 30% of the time for hard combinations. That's fine — you only need the successes for training. Generate 8 samples, keep the best verified one. The verifiers make this efficient because checks are instant (no human review needed).

#### Step 4: Balance the Dataset

After generation, **audit** the constraint distribution:

```python
def audit_dataset(examples: List[InstructionExample]) -> Dict:
    category_counts = Counter()
    subcategory_counts = Counter()
    difficulty_dist = Counter()
    
    for ex in examples:
        for c in ex.constraints:
            category_counts[c.category] += 1
            subcategory_counts[c.subcategory] += 1
        difficulty_dist[len(ex.constraints)] += 1
    
    return {
        "total": len(examples),
        "by_category": dict(category_counts),
        "by_subcategory": dict(subcategory_counts),
        "by_num_constraints": dict(difficulty_dist),
        "gaps": identify_gaps(subcategory_counts),
    }
```

A well-balanced instruction-following dataset should have the following approximate distribution:

| Category | Target % | Rationale |
|----------|---------|-----------|
| Format constraints | 25-30% | Most practically important, high failure rate |
| Length constraints | 15-20% | Common in production, verifiable |
| Content inclusion/exclusion | 15-20% | Critical for safety-adjacent tasks |
| Structural constraints | 10-15% | Table, list, section structure |
| Style/tone constraints | 10-15% | Important but harder to verify |
| Language/locale | 5-10% | Critical for multilingual deployment |
| Multi-constraint (3+) | 10-15% | Explicitly exercises composition |

### 4.4 Negative Examples and Contrastive Training

Positive-only training teaches the model what *right* looks like. But for instruction following, **contrastive data** — showing what *wrong* looks like — is equally valuable.

#### Approach 1: Near-Miss Negatives for DPO/RLHF

Generate pairs where the chosen response satisfies all constraints and the rejected response violates exactly one:

```python
def create_contrastive_pair(
    example: InstructionExample,
    model: str
) -> Tuple[str, str]:
    """
    Create (chosen, rejected) pair where rejected violates one constraint.
    """
    chosen = example.response  # Already verified
    
    # Generate a response with one constraint removed from the prompt
    constraint_to_violate = random.choice(example.constraints)
    reduced_prompt = remove_constraint(example.prompt, constraint_to_violate)
    
    rejected_candidates = [
        generate(model, reduced_prompt) for _ in range(4)
    ]
    
    # Find one that violates the removed constraint but satisfies others
    for candidate in rejected_candidates:
        satisfaction = verify_all_constraints(candidate, example.constraints)
        if (not satisfaction[constraint_to_violate.subcategory] and
            all(v for k, v in satisfaction.items() 
                if k != constraint_to_violate.subcategory and v is not None)):
            return chosen, candidate
    
    return None  # Couldn't find a good near-miss
```

**Why near-miss negatives?** If the rejected response is terrible in every way, the model learns nothing about instruction following — it just learns to be generally better. A near-miss that's perfect *except* for format teaches the model the specific importance of format compliance.

#### Approach 2: Constraint-Violation Augmentation

Take a good response and systematically break one constraint:

```python
VIOLATION_FUNCTIONS = {
    "word_count_exact": lambda resp, n: truncate_or_extend(resp, n + 15),
    "all_uppercase": lambda resp, _: resp.lower(),  
    "valid_json": lambda resp, _: corrupt_json(resp),
    "bullet_count": lambda resp, n: add_extra_bullets(resp, n + 2),
    "starts_with": lambda resp, prefix: "Well, " + resp,
    "language_check": lambda resp, lang: mix_in_english(resp),
}

def create_augmented_negative(
    response: str, 
    constraint: Constraint
) -> str:
    if constraint.subcategory in VIOLATION_FUNCTIONS:
        return VIOLATION_FUNCTIONS[constraint.subcategory](
            response, constraint.specification
        )
    return None
```

### 4.5 RL-Based Approaches: Reward Shaping for Instruction Following

For GRPO/PPO post-training, design rewards that explicitly measure constraint satisfaction:

```python
def instruction_following_reward(
    prompt: str, 
    response: str, 
    constraints: List[Constraint]
) -> float:
    satisfaction = verify_all_constraints(response, constraints)
    
    # Per-constraint scores
    n_verifiable = sum(1 for v in satisfaction.values() if v is not None)
    n_satisfied = sum(1 for v in satisfaction.values() if v is True)
    
    if n_verifiable == 0:
        return 0.0
    
    # Base score: fraction of constraints satisfied
    constraint_score = n_satisfied / n_verifiable
    
    # Bonus for full satisfaction (all-or-nothing incentive)
    full_bonus = 0.2 if n_satisfied == n_verifiable else 0.0
    
    # Quality score (separate from constraint satisfaction)
    quality_score = assess_response_quality(prompt, response)
    
    # Combined reward
    return 0.5 * constraint_score + 0.2 * full_bonus + 0.3 * quality_score
```

**Key design choice: the all-or-nothing bonus.** Without it, a model that satisfies 3/4 constraints gets 75% reward, which is acceptable. With the bonus, full satisfaction is noticeably more rewarding than partial, training the model to *try harder* on the last constraint rather than optimizing for the easy ones.

### 4.6 The RECAST Approach: Scaling Multi-Constraint Data

RECAST (2024) provides a principled framework for scaling constrained instruction-following data:

1. **Extract constraints from real prompts:** Mine production logs for user prompts that contain explicit constraints. Categorize and decompose them.
2. **Recombine constraints synthetically:** Take extracted constraint types and create new combinations that were never seen in the wild — this forces generalization.
3. **Difficulty curriculum:** Start training with single-constraint examples, then gradually introduce multi-constraint compositions.

```python
def curriculum_schedule(epoch: int, total_epochs: int) -> Dict[str, float]:
    """
    Gradually increase constraint difficulty during training.
    """
    progress = epoch / total_epochs
    
    if progress < 0.3:
        return {"1_constraint": 0.7, "2_constraints": 0.25, "3+_constraints": 0.05}
    elif progress < 0.7:
        return {"1_constraint": 0.3, "2_constraints": 0.4, "3+_constraints": 0.3}
    else:
        return {"1_constraint": 0.1, "2_constraints": 0.3, "3+_constraints": 0.6}
```

### 4.7 Domain-Specific Instruction Following

Generic instruction-following training helps, but production use cases often require **domain-specific** constraint types:

| Domain | Custom Constraint Types | Example |
|--------|------------------------|---------|
| **API/Tool Use** | Valid JSON Schema, parameter types, enum values | "Return a function call with exactly these parameters" |
| **Code Generation** | Syntax validity, language-specific, includes tests | "Write Python 3.10+ code with type hints" |
| **Content Writing** | Brand voice, SEO keywords, readability score | "Include 3 target keywords, Flesch score > 60" |
| **Customer Support** | Empathy markers, escalation protocol, no promises | "Acknowledge frustration, offer max 2 solutions" |
| **Medical/Legal** | Citation requirements, disclaimer placement, hedging | "Include 'this is not medical advice' disclaimer" |

**For each domain, build domain-specific verifiers:**

```python
# Example: API/Tool Use verifiers
API_VERIFIERS = {
    "valid_function_call": lambda resp, schema: validate_against_schema(
        extract_function_call(resp), schema
    ),
    "correct_param_types": lambda resp, schema: check_param_types(
        extract_function_call(resp), schema
    ),
    "no_hallucinated_tools": lambda resp, tool_list: all(
        tool in tool_list for tool in extract_tool_names(resp)
    ),
}
```

### 4.8 Concrete Training Examples

Abstract pipeline code is useful, but **seeing exact prompt-response pairs** is what makes training data design concrete. Below are complete examples across every constraint category — ready to adapt for your own data pipeline.

---

#### Category 1: Format Constraints

**Example 1a — JSON Output (SFT)**

**Prompt:**
```
List 3 benefits of remote work. Respond ONLY with a valid JSON array where each
element has "benefit" (string) and "impact_level" (one of: "high", "medium", "low").
No text outside the JSON.
```

**✅ Correct Response:**
```json
[
  {"benefit": "Eliminates commute time, giving employees 1-2 extra hours daily", "impact_level": "high"},
  {"benefit": "Reduces office overhead costs for employers", "impact_level": "medium"},
  {"benefit": "Enables hiring from a broader geographic talent pool", "impact_level": "high"}
]
```

**❌ Common Failure (use as DPO rejected):**
```
Here are 3 benefits of remote work:

1. **No commute** - Saves 1-2 hours daily (high impact)
2. **Lower costs** - Reduces office expenses (medium impact)
3. **Wider talent pool** - Can hire from anywhere (high impact)
```
*Failure: returns markdown prose instead of JSON. Content is correct but format constraint is completely ignored.*

---

**Example 1b — Markdown Table Format (SFT)**

**Prompt:**
```
Compare Python and JavaScript. Present your answer as a markdown table with exactly
these columns: Feature, Python, JavaScript. Include exactly 4 rows.
```

**✅ Correct Response:**
```
| Feature | Python | JavaScript |
|---------|--------|------------|
| Typing | Dynamically typed with optional type hints | Dynamically typed with TypeScript as typed superset |
| Primary Use | Data science, ML, backend services, scripting | Web frontends, Node.js backends, full-stack apps |
| Package Manager | pip with PyPI repository | npm with npmjs.com registry |
| Concurrency | GIL limits true threading; asyncio for I/O | Event loop with non-blocking I/O by default |
```

---

**Example 1c — All Uppercase (SFT)**

**Prompt:**
```
Explain what an API is in one paragraph. YOUR ENTIRE RESPONSE MUST BE IN UPPERCASE.
```

**✅ Correct Response:**
```
AN API, OR APPLICATION PROGRAMMING INTERFACE, IS A SET OF RULES AND PROTOCOLS THAT
ALLOWS DIFFERENT SOFTWARE APPLICATIONS TO COMMUNICATE WITH EACH OTHER. IT DEFINES
THE METHODS AND DATA FORMATS THAT PROGRAMS CAN USE TO REQUEST AND EXCHANGE
INFORMATION. FOR EXAMPLE, WHEN A WEATHER APP ON YOUR PHONE DISPLAYS THE CURRENT
TEMPERATURE, IT USES AN API TO FETCH THAT DATA FROM A REMOTE SERVER. APIS ENABLE
DEVELOPERS TO BUILD ON EXISTING SERVICES WITHOUT NEEDING TO UNDERSTAND THEIR
INTERNAL IMPLEMENTATION, MAKING SOFTWARE DEVELOPMENT FASTER AND MORE MODULAR.
```

**❌ Near-miss Failure (DPO rejected):**
```
AN API, OR APPLICATION PROGRAMMING INTERFACE, IS A SET OF RULES THAT ALLOWS
SOFTWARE APPLICATIONS TO COMMUNICATE. It defines methods and data formats for
requesting information. For example, weather apps use APIs to fetch temperature
data from servers. APIs make development faster and more modular.
```
*Failure: starts uppercase but drifts to normal casing after the first sentence — a classic instruction dilution failure.*

---

#### Category 2: Length Constraints

**Example 2a — Exact Word Count (SFT)**

**Prompt:**
```
What is machine learning? Answer in exactly 30 words.
```

**✅ Correct Response (30 words):**
```
Machine learning is a branch of artificial intelligence where computers learn
patterns from data to make predictions or decisions without being explicitly
programmed for each specific task.
```

**❌ Failure — 47 words (DPO rejected):**
```
Machine learning is a subset of artificial intelligence that enables computers to
learn from data and improve their performance over time without being explicitly
programmed. It involves training algorithms on large datasets to recognize patterns
and make predictions or decisions based on new, unseen data.
```
*Failure: content is accurate but exceeds the 30-word constraint by 57%.*

---

**Example 2b — Sentence Count (SFT)**

**Prompt:**
```
Describe the water cycle. Use exactly 3 sentences.
```

**✅ Correct Response:**
```
The water cycle begins when the sun heats surface water in oceans, lakes, and
rivers, causing it to evaporate into the atmosphere as water vapor. This vapor
rises, cools, and condenses into clouds through a process called condensation,
forming tiny water droplets or ice crystals. Eventually, these droplets grow
heavy enough to fall back to Earth as precipitation — rain, snow, or hail — where
the water collects and the cycle begins again.
```

---

**Example 2c — Min/Max Word Count Range (SFT)**

**Prompt:**
```
Explain what a database index is. Your response must be between 40 and 60 words.
```

**✅ Correct Response (52 words):**
```
A database index is a data structure that improves the speed of data retrieval
operations on a table. It works like a book's index — instead of scanning every
row, the database looks up the index to find the exact location of the desired
data, dramatically reducing query time for large datasets.
```

---

#### Category 3: Content Inclusion/Exclusion

**Example 3a — Must Include Phrase (SFT)**

**Prompt:**
```
Write a short motivational message for someone starting a new job. You must include
the exact phrase "every expert was once a beginner" somewhere in your response.
```

**✅ Correct Response:**
```
Starting a new role can feel overwhelming, but remember that every expert was once
a beginner. The uncertainty you feel today is the same feeling that every successful
person in your field once experienced. Give yourself permission to ask questions,
make mistakes, and learn at your own pace. Six months from now, you'll look back
and be amazed at how far you've come.
```

---

**Example 3b — Must Exclude Topic (SFT)**

**Prompt:**
```
Write a 3-sentence review of a coffee shop. Do NOT mention the price, cost, or
value of anything.
```

**✅ Correct Response:**
```
The espresso at Ember & Oak has a rich, chocolatey flavor with a smooth finish that
rivals any specialty roaster in the city. The interior blends exposed brick with
warm lighting, creating a cozy atmosphere perfect for focused work or quiet
conversation. Their baristas clearly take pride in their craft, pulling each shot
with precision and serving every drink with a genuine smile.
```

**❌ Near-miss Failure (DPO rejected):**
```
The espresso at Ember & Oak has a rich, chocolatey flavor that's well worth the
$4.50 price tag. The interior blends exposed brick with warm lighting, creating a
cozy atmosphere. Their baristas take pride in their craft, making it a great value
for specialty coffee lovers.
```
*Failure: mentions "$4.50 price tag" and "great value" — directly violating the exclusion constraint.*

---

**Example 3c — Required Opening (SFT)**

**Prompt:**
```
What is photosynthesis? Begin your response with the word "Fundamentally" and end
your response with the word "energy."
```

**✅ Correct Response:**
```
Fundamentally, photosynthesis is the process by which green plants and certain
other organisms convert light into chemical fuel. Chlorophyll in plant cells
absorbs sunlight and uses it to transform carbon dioxide from the air and water
from the soil into glucose and oxygen. This glucose serves as the plant's primary
source of energy.
```

---

#### Category 4: Structural Constraints

**Example 4a — Numbered List with Specific Count (SFT)**

**Prompt:**
```
Give me tips for public speaking. Respond as a numbered list with exactly 5 items.
Each item must be one sentence only.
```

**✅ Correct Response:**
```
1. Practice your opening 10 times until you can deliver it without thinking, because a strong start builds confidence for everything that follows.
2. Make eye contact with one person at a time for 3-5 seconds rather than scanning the room, which feels more natural for both you and the audience.
3. Pause deliberately after key points instead of filling silence with "um" or "uh," as these pauses actually make you sound more authoritative.
4. Arrive early to walk the stage and test the microphone so the physical space feels familiar before the audience enters.
5. Record yourself during rehearsal and watch it back, focusing on body language habits you would never notice in the moment.
```

---

**Example 4b — Section Structure (SFT)**

**Prompt:**
```
Write about the programming language Rust. Organize your response into exactly
three sections with these headers: "Overview", "Key Strengths", "When to Use It".
Keep each section to 2-3 sentences.
```

**✅ Correct Response:**
```
## Overview

Rust is a systems programming language developed by Mozilla Research, first released
in 2015. It focuses on performance, reliability, and memory safety without requiring
a garbage collector.

## Key Strengths

Rust's ownership model catches memory bugs at compile time, eliminating entire
classes of vulnerabilities like buffer overflows and data races. Its performance
matches C and C++ while providing modern language features like pattern matching,
algebraic data types, and a powerful macro system. The compiler's strict checks
produce a "if it compiles, it works" experience that reduces debugging time.

## When to Use It

Rust excels in performance-critical systems like operating systems, game engines,
embedded devices, and WebAssembly modules. It's also increasingly adopted for
backend web services where both speed and reliability matter.
```

---

#### Category 5: Style/Tone Constraints

**Example 5a — Explain Like I'm 5 (SFT)**

**Prompt:**
```
Explain how the internet works. Use language that a 5-year-old would understand. No
technical jargon at all.
```

**✅ Correct Response:**
```
Imagine you want to send a drawing to your friend who lives far away. You put the
drawing in an envelope and a mail carrier takes it through roads until it gets to
your friend's house. The internet works kind of like that, but instead of drawings,
you're sending messages, pictures, and videos. And instead of a mail carrier walking
on roads, the messages zoom through special wires really, really fast — so fast that
your friend gets your message almost right away, even if they live on the other
side of the world!
```

---

**Example 5b — Formal Academic Tone (SFT)**

**Prompt:**
```
Discuss the impact of social media on political discourse. Use formal academic
language throughout. Do not use contractions.
```

**✅ Correct Response:**
```
The proliferation of social media platforms has fundamentally altered the landscape
of political discourse in democratic societies. These platforms have democratized
the dissemination of political information, enabling citizens to engage directly
with elected officials and policy debates without the intermediation of traditional
media gatekeepers. However, this democratization has not been without consequence.

The algorithmic curation of content has been demonstrated to reinforce ideological
echo chambers, wherein users are disproportionately exposed to viewpoints that
align with their existing beliefs. This phenomenon, frequently referred to as
"filter bubbles," has been associated with increased political polarization and a
diminished capacity for cross-partisan dialogue. Furthermore, the velocity at which
information propagates through these networks has complicated efforts to distinguish
verified reporting from misinformation, presenting significant challenges for the
integrity of public deliberation.
```

**❌ Near-miss Failure (DPO rejected):**
```
Social media has fundamentally changed how we talk about politics. These platforms
let people engage directly with politicians and debates without traditional media
acting as gatekeepers. But it's not all positive.

Algorithms create echo chambers where you only see viewpoints you already agree with.
This has increased polarization and made it harder to have productive conversations
across party lines. And the speed of information makes it tough to tell what's real
from what's not.
```
*Failure: uses contractions ("it's", "what's"), casual phrases ("But it's not all positive"), and second person ("you") — violating the formal academic tone constraint.*

---

#### Category 6: Language/Locale Constraints

**Example 6a — Respond in Specific Language (SFT)**

**Prompt:**
```
What are the three branches of the United States government? Respond entirely in
Spanish. Do not include any English words.
```

**✅ Correct Response:**
```
Las tres ramas del gobierno de los Estados Unidos son:

1. **Poder Legislativo** — El Congreso, compuesto por el Senado y la Cámara de
Representantes, es responsable de crear las leyes.

2. **Poder Ejecutivo** — Encabezado por el Presidente, se encarga de implementar y
hacer cumplir las leyes aprobadas por el Congreso.

3. **Poder Judicial** — La Corte Suprema y los tribunales federales interpretan las
leyes y determinan si son compatibles con la Constitución.

Este sistema de separación de poderes fue diseñado para evitar la concentración de
autoridad en una sola rama del gobierno.
```

---

**Example 6b — Date and Number Format (SFT)**

**Prompt:**
```
List 3 historical events with their dates. Use European date format (DD/MM/YYYY) and
European number formatting (dots for thousands, commas for decimals) for any numbers.
```

**✅ Correct Response:**
```
1. **Moon Landing** (20/07/1969) — An estimated 600.000.000 people worldwide watched
as astronauts first set foot on the lunar surface, an event that cost approximately
25.400.000.000 dollars.

2. **Fall of the Berlin Wall** (09/11/1989) — After 28 years of division, approximately
5.000 East Germans crossed into West Berlin on the first night alone.

3. **World Wide Web Goes Public** (06/08/1991) — Tim Berners-Lee published the first
website. By the end of the decade, there were roughly 17.000.000 websites online.
```

---

#### Category 7: Multi-Constraint Compositions

**Example 7a — Format + Length + Content (SFT, Hard)**

**Prompt:**
```
Recommend a book for learning Python. Your response must satisfy ALL of these:
- Respond in valid JSON with keys: "title", "author", "reason" (string), "level"
  (one of: "beginner", "intermediate", "advanced")
- The "reason" value must be between 15 and 25 words
- Include the phrase "hands-on" somewhere in your response
- No text outside the JSON object
```

**✅ Correct Response:**
```json
{
  "title": "Python Crash Course",
  "author": "Eric Matthes",
  "reason": "Excellent hands-on introduction that builds real projects like games and web apps while teaching core programming concepts clearly",
  "level": "beginner"
}
```
*Verification: valid JSON ✓ | "reason" is 18 words ✓ | contains "hands-on" ✓ | no text outside JSON ✓*

---

**Example 7b — Style + Structure + Exclusion (SFT, Hard)**

**Prompt:**
```
Explain cloud computing. Your response must:
- Be written as exactly 4 bullet points
- Use a casual, conversational tone
- Not contain any questions
- Each bullet point must start with a verb
```

**✅ Correct Response:**
```
- Think of cloud computing as renting someone else's super-powerful computer instead of buying your own — you just use what you need and pay as you go.
- Store your files, run your apps, and crunch your data on servers that live in massive data centers around the world, accessible from anywhere with an internet connection.
- Scale up instantly when your app goes viral, then scale back down when things quiet off — no need to buy hardware that sits idle 90% of the time.
- Save yourself the headache of maintaining physical servers, dealing with hardware failures, and hiring a whole IT team just to keep the lights on.
```
*Verification: 4 bullets ✓ | casual tone ✓ | no questions ✓ | each starts with a verb (Think, Store, Scale, Save) ✓*

---

**Example 7c — Length + Language + Structure + Format (SFT, Adversarial)**

**Prompt:**
```
List the benefits of exercise. ALL of these constraints must be met:
- Respond entirely in French
- Use exactly 3 numbered items
- Each item must be exactly one sentence
- Total response must be between 40 and 70 words
- Write entirely in ALL CAPS
```

**✅ Correct Response:**
```
1. L'EXERCICE RÉGULIER RENFORCE LE SYSTÈME CARDIOVASCULAIRE ET RÉDUIT CONSIDÉRABLEMENT LE RISQUE DE MALADIES CHRONIQUES COMME LE DIABÈTE ET L'HYPERTENSION.
2. L'ACTIVITÉ PHYSIQUE LIBÈRE DES ENDORPHINES QUI AMÉLIORENT L'HUMEUR ET DIMINUENT LES SYMPTÔMES D'ANXIÉTÉ ET DE DÉPRESSION.
3. FAIRE DU SPORT RÉGULIÈREMENT FAVORISE UN SOMMEIL PLUS PROFOND ET UNE MEILLEURE RÉCUPÉRATION PHYSIQUE ET MENTALE QUOTIDIENNE.
```
*Verification: French ✓ | 3 numbered items ✓ | each one sentence ✓ | 49 words ✓ | all caps ✓*

---

#### DPO Contrastive Pairs Collection

The examples above already include some chosen/rejected pairs. Here's a systematic set designed specifically for DPO training, where the **rejected response is a near-miss** — high quality content but violating exactly one constraint:

**DPO Pair 1 — Word count violation only:**

| | Prompt | Chosen | Rejected |
|---|--------|--------|----------|
| **Constraint** | "Define gravity in exactly 20 words" | "Gravity is the natural force that attracts objects with mass toward each other, keeping planets in orbit and us on Earth." (20 words) | "Gravity is a fundamental force of nature that pulls objects with mass toward each other. It keeps planets in orbit around stars and holds us firmly on the surface of the Earth." (33 words) |
| **What differs** | — | All constraints met | Content equally good, but word count violated |

**DPO Pair 2 — JSON validity violation only:**

| | Prompt | Chosen | Rejected |
|---|--------|--------|----------|
| **Constraint** | "Name 2 planets as a JSON array of objects with 'name' and 'position' keys" | `[{"name": "Mars", "position": 4}, {"name": "Jupiter", "position": 5}]` | `[{name: "Mars", position: 4}, {name: "Jupiter", position: 5}]` |
| **What differs** | — | Valid JSON | Missing quotes around keys — invalid JSON |

**DPO Pair 3 — Exclusion constraint violation only:**

| | Prompt | Chosen | Rejected |
|---|--------|--------|----------|
| **Constraint** | "Describe our product's benefits. Do not mention competitors by name." | "Our platform processes data 10x faster through proprietary streaming architecture, enabling real-time analytics that were previously impossible." | "Our platform processes data 10x faster than Snowflake through proprietary streaming architecture, enabling real-time analytics that competitors like Databricks still struggle with." |
| **What differs** | — | No competitor names | Mentions "Snowflake" and "Databricks" |

**DPO Pair 4 — Persona drift:**

| | Prompt | Chosen | Rejected |
|---|--------|--------|----------|
| **Constraint** | "Explain compound interest as a pirate would. Maintain the pirate persona throughout." | "Arrr, listen up, ye landlubber! Compound interest be like buryin' yer treasure and watchin' it grow on its own, it does. Each year, yer gold earns more gold — and then THAT gold earns gold too! Before ye know it, yer tiny chest of doubloons becomes a mountain o' riches. The secret, savvy? Start buryin' yer treasure early, and let the seas of time do the work for ye!" | "Arrr, compound interest be like buried treasure that grows! Each year, your money earns interest, and then that interest also earns interest. Over time, this creates exponential growth. For example, $1,000 at 7% annual compound interest becomes $1,967 after 10 years. The earlier you start investing, the more powerful this effect becomes." |
| **What differs** | — | Maintains pirate voice throughout | Drops persona after first sentence, switches to textbook tone |

---

> **How to use these examples:** These prompt-response pairs can be directly used as SFT training data (positive examples only) or as DPO preference pairs (chosen + rejected). For RL (GRPO/PPO), use the prompts with the constraint verifiers from Section 4.2 as the reward function — the model generates its own responses and the verifiers score them. The exact examples above also serve as **unit tests for your verifier pipeline**: run each ✅ response through your verifiers to confirm they pass, and each ❌ response to confirm they fail.

---

## 5. Evaluation: Measuring What You Built

### 5.1 IFEval and Beyond

| Benchmark | Focus | Constraints | Limitation |
|-----------|-------|-------------|------------|
| **IFEval** | Verifiable format/length constraints | ~25 types, 500 prompts | Synthetic constraints, not representative of production |
| **WildIFEval** | Real-world multi-constraint prompts | Mined from user interactions | Smaller scale |
| **IFEval-FC** | Function calling format constraints | Schema adherence, date formats, casing | Narrow to API use cases |
| **MM-IFEval** | Multimodal instruction following | 6 categories, 32 subcategories | Requires multimodal models |
| **Inverse IFEval** | Counter-intuitive instructions | 8 categories including "deliberately wrong" | Tests cognitive flexibility, not typical use |

### 5.2 Custom Evaluation Protocol

For production systems, build an eval suite that mirrors your actual constraint distribution:

```python
def evaluate_instruction_following(
    model,
    eval_set: List[InstructionExample],
    metrics: List[str] = ["strict", "loose", "per_category"]
) -> Dict:
    results = {"strict_pass": 0, "loose_pass": 0, "per_category": defaultdict(list)}
    
    for example in eval_set:
        response = model.generate(example.prompt)
        satisfaction = verify_all_constraints(response, example.constraints)
        
        all_satisfied = all(v for v in satisfaction.values() if v is not None)
        any_satisfied = any(v for v in satisfaction.values() if v is not None)
        
        results["strict_pass"] += int(all_satisfied)
        results["loose_pass"] += int(any_satisfied)
        
        for constraint in example.constraints:
            results["per_category"][constraint.category].append(
                satisfaction.get(constraint.subcategory, None)
            )
    
    # Compute rates
    n = len(eval_set)
    results["strict_rate"] = results["strict_pass"] / n
    results["loose_rate"] = results["loose_pass"] / n
    
    for cat, values in results["per_category"].items():
        valid = [v for v in values if v is not None]
        results["per_category"][cat] = sum(valid) / len(valid) if valid else 0
    
    return results
```

### 5.3 What to Measure

| Metric | Definition | Target |
|--------|-----------|--------|
| **Strict Accuracy** | % of prompts where ALL constraints are satisfied | > 80% |
| **Per-Constraint Accuracy** | % satisfaction rate per constraint type | > 90% per type |
| **Multi-Constraint Decay** | How accuracy drops as # constraints increases | < 10% drop per added constraint |
| **Persistence** | Constraint satisfaction at token position 500 vs. 50 | < 5% drop |
| **Quality Trade-off** | Response quality when constraints are enforced vs. unconstrained | < 10% quality loss |

---

## 6. Practical Recommendations

### For teams starting from scratch:

1. **Start with format and length constraints** — they're the most verifiable and highest-impact in production
2. **Use rejection sampling over a teacher model** — generate N, verify, keep the best. This is far more efficient than human annotation
3. **Build verifiers first, data second** — your verifier quality determines your data quality which determines your model quality
4. **Target 5,000-10,000 verified examples** covering all seven constraint categories — quality beats quantity
5. **Include 10-15% multi-constraint examples** from day one — don't defer composition to later

### For teams with existing SFT data:

1. **Audit your constraint coverage** — most SFT datasets are heavily biased toward content quality and almost empty on format/length constraints
2. **Add contrastive pairs** for DPO — near-miss negatives that violate exactly one constraint are the highest-value data you can create
3. **Use a difficulty curriculum** — start with single constraints, ramp to multi-constraint
4. **Monitor per-category regressions** — improving JSON compliance shouldn't break word count adherence

### Common mistakes:

- **Too many easy examples:** 10,000 "use bullet points" examples teach less than 500 "exactly 3 bullet points, each under 15 words, in formal tone"
- **No verification pipeline:** Unverified training examples introduce noise that actively hurts — the model learns that "roughly following" constraints is acceptable
- **Ignoring the quality trade-off:** Optimizing purely for constraint satisfaction can produce stilted, robotic responses. Always measure *both* constraint adherence and response quality
- **Single-constraint only:** Models trained only on individual constraints fail catastrophically at compositions. Explicitly train on multi-constraint examples

---

## 7. Open Questions

1. **Constraint-aware decoding:** Can we enforce constraints at inference time (e.g., tracking word count during generation) without sacrificing quality? Early results with constrained decoding are promising but add latency.

2. **Learning to plan ahead:** Autoregressive models struggle with global constraints (total word count, balanced structure). Can we teach models to outline first, then fill in? Chain-of-thought for instruction planning.

3. **Constraint difficulty prediction:** Given a constraint set, can we predict how likely the model is to satisfy all constraints? This would enable dynamic routing — hard constraint sets go to stronger models.

4. **Cross-lingual transfer:** Does instruction-following training in English transfer to other languages? Early evidence from M-IFEval suggests partial transfer, but language-specific constraints (date formats, honorifics) require language-specific data.

5. **Constraint composition scaling laws:** How does joint satisfaction rate scale with the number of constraints? Is the multiplicative independence assumption correct, or do models learn compositional strategies that beat it?

---

## References

### Benchmarks
1. **IFEval** — Zhou et al., 2023 — Instruction-Following Eval with ~25 verifiable constraint types
2. **WildIFEval** — 2024 — Real-world multi-constraint instruction evaluation
3. **IFEval-FC** — 2024 — Function calling format compliance benchmark
4. **MM-IFEval** — 2024 — Multimodal instruction following (6 categories, 32 subcategories)
5. **M-IFEval** — 2024 — Multilingual instruction following (French, Japanese, Spanish)
6. **Inverse IFEval** — 2024 — Counter-intuitive instruction following (cognitive inertia)

### Training Methods
7. **RECAST** — 2024 — Constraint extraction and recombination for multi-constraint SFT data
8. **Self-Instruct** — Wang et al., 2023 — LLM-generated instruction-following data
9. **GLAN** — 2024 — Generalized instruction tuning with synthetic data
10. **Conifer** — 2024 — Instruction tuning with complex multi-constraint instructions

### Analysis
11. **"The Instruction Gap"** — 2024 — Enterprise instruction-following failure analysis
12. **"Reasoning Hurts Instruction Following"** — Amazon Science, 2024 — CoT degrades constraint adherence
13. **"Instructional Distraction"** — ACL 2024 — Models confuse input text with instructions
14. **"Lost in the Middle"** — Liu et al., 2023 — Attention degradation in long contexts

*Code examples are synthesized implementations illustrating practical patterns.*
