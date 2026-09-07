---
author: Jing Lu
pubDatetime: 2026-03-11T00:00:00Z
modDatetime: 2026-09-06T00:00:00-07:00
title: "Improving LLM Internationalization: Bridging the Gap in Tool Use and Agency"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Agents
  - i18n
description: "A practical multilingual agent playbook, updated with task-specific evidence from MASSIVE-Agents, GAIA-v2-LILT, and SEATauBench, with explicit limits on what transfers to current frontier models."
---

**Multilingual agent performance is uneven, but the gap depends on the model, task, and what has been localized.** In the 2025 [MASSIVE-Agents study](https://aclanthology.org/2025.findings-emnlp.1099/), Nova Premier scored 57.37% English AST accuracy, 34.05% across 52 languages, and 6.81% in Amharic. These are historical static function-call results, not a current frontier-agent success rate.

This post gives engineering options for tool use and agentic systems. Treat the architecture and training suggestions as hypotheses to test on your deployment, rather than universal improvements.

## September 2026 update: what later agent research changes

Two papers published after this post sharpen its original argument:

| New evidence                                                 | What it changes                                                                                                                                                                           | What to test                                                                     |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [GAIA-v2-LILT, April 2026](https://arxiv.org/abs/2604.24929) | Re-auditing translated tasks improves agent scores. Some apparent language gaps come from answer alignment, cultural context, and difficulty changes. Residual gaps remain.               | Audit the task and answer key before attributing a failure to the language.      |
| [SEATauBench, June 2026](https://arxiv.org/abs/2606.28715)   | Localizing the dialogue is a weaker stress test than localizing policies, tools, and the full business context. Agent and simulated-user behavior both contribute to the measured result. | Compare dialogue-only, tool-localized, and fully localized workflows separately. |

The practical change is to evaluate **completed actions and final state**, alongside tool syntax and language fluency. Keep model versions, budgets, tasks, and simulator settings visible. Neither study provides a complete ranking of today's frontier models across every language.

For the detailed language-by-task results and their limitations, see the [30-language visual evidence review](/posts/2026-09-05-multilingual-agent-language-gaps/).

---

## 1. Why Tool Use Adds i18n Failure Points

Translation, QA, static function calling, and interactive task completion use different metrics. Their gaps cannot be compared with one universal multiplier. Tool use adds several places where a language mismatch can affect the final action.

Agentic tool use requires the model to execute a chain of four capabilities **simultaneously**, each compounding the multilingual difficulty:

| Step                        | What the Model Must Do                           | i18n-Specific Challenge                                                            |
| --------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------- |
| 1. **Intent Understanding** | Parse the user's request                         | Handling morphologically complex, low-resource, or code-switched input             |
| 2. **Tool Selection**       | Choose the right function/API                    | Tool descriptions are almost always in English; cross-lingual retrieval degrades   |
| 3. **Parameter Extraction** | Generate structured JSON arguments               | Non-Latin names, locale-specific formats (dates, currencies), token inflation      |
| 4. **Output Synthesis**     | Interpret results and respond in user's language | Translating tool outputs while preserving technical precision and cultural context |

As an illustrative calculation, four required stages with 85% success each would yield $0.85^4 \approx 52\%$ if the stage successes were independent. At 70%, the product is about 24%. These are hypothetical rates, not language measurements; correlated errors and recovery steps change the result.

**Tokenization** can amplify the challenge. Measure tokens for semantically matched requests using the deployed tokenizer. The ratio varies by language, script, content, and tokenizer; there is no universal 3–5× factor. Count localized content separately from tool schemas that remain in English.

---

## 2. Training-Time Approaches

### 2.1 Pre-Training: Getting the Foundation Right

| Strategy                        | Mechanism                                                                                          | Impact                                                                                        |
| ------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Dynamic Data Sampling**       | Over-sample low-resource languages during pre-training to compensate for web data imbalance        | A candidate for reducing underrepresentation; validate per language                           |
| **Language-Aware Tokenization** | Design tokenizers sensitive to morphological differences (agglutinative languages, CJK characters) | Can improve token efficiency; measure the deployed tokenizer                                  |
| **Mixture of Experts (MoE)**    | Route different languages to specialized expert sub-networks                                       | Mitigates the "curse of multilinguality" — performance interference between distant languages |
| **Cross-Lingual Embeddings**    | Learn unified representations across languages in a shared semantic space                          | Enables zero-shot transfer to unseen languages                                                |

The **curse of multilinguality** deserves special attention. When a single dense model is trained on 100+ languages, adding more languages eventually _degrades_ performance on existing ones — the model's capacity gets spread too thin. MoE adds conditional capacity, but learned experts need not correspond to language families and do not guarantee the interference disappears.

### 2.2 Post-Training: Targeted i18n Fine-Tuning

Post-training is where the most actionable improvements happen, because you get to control the data precisely:

**Cross-Lingual Optimization (CLO)** — The most data-efficient approach: transfer English-centric LLMs to target languages using English instruction data + a translation model. CLO is significantly more data-efficient than standard SFT in low-resource settings because it leverages the model's existing English capabilities as a bridge.

**Multilingual RLHF** — A critical and often overlooked step. When RLHF is conducted only on English preference data, the resulting reward model learns "English quality = universal quality." This creates a bias where the model generates anglicized responses even in non-English languages. The fix: collect preference judgments from native speakers across target languages and train separate or multi-headed reward models.

**Synthetic Multilingual Data** — Use multiple "teacher" LLMs, each strong in different languages, to generate high-quality synthetic SFT data. A Gemini model strong in Japanese generates Japanese instruction-response pairs; a model strong in Arabic generates Arabic pairs. The student model trains on the combined multilingual corpus.

**TransLLM Framework** — Breaks English→Target transfer into sub-tasks via "translation chain-of-thought" + LoRA + recovery knowledge distillation. The key innovation is preventing **catastrophic forgetting** — when a model fine-tuned for a target language loses its English capabilities and general reasoning ability.

**Model Merging** — Merge weights from language-specific fine-tuned models for cross-lingual transfer. Especially useful in low-resource settings where you don't have enough data for standard SFT but can merge a Korean-specialized model with a tool-use-specialized model.

### 2.3 Multilingual Tool-Use Data: The State of the Art

Three earlier resources are useful starting points; their tasks and model cohorts differ:

#### NAACL 2025: Enhancing Function-Calling Capabilities

[Chen et al.](https://aclanthology.org/2025.naacl-industry.9/) showed that a **tailored translation pipeline** for function-calling data significantly improves non-English tool use, with particularly strong results for Traditional Chinese. The key insight: instruction-following data enhances both function-calling accuracy **and** relevance detection — the ability to know when _not_ to call a tool.

Simply translating English tool-calling datasets doesn't work well. The translation must preserve the structural relationship between the user query, the function signature, and the expected arguments, which generic machine translation handles poorly.

#### EMNLP 2025: MASSIVE-Agents Benchmark

The [paper's Table 2](https://aclanthology.org/2025.findings-emnlp.1099.pdf) reports Nova Premier's zero-shot AST results on the 10k dataset split (9,741 items total, including 190 English and 191 Amharic items):

| Model and setting            | English | Average, 52 languages | Amharic |
| ---------------------------- | ------: | --------------------: | ------: |
| Nova Premier, zero-shot, 10k |  57.37% |                34.05% |   6.81% |

The full converted benchmark has 47,020 samples, averaging about 904 per language; it does not contain exactly 904 samples in every language. It covers 55 functions and 286 arguments. AST matching tests function and argument structure rather than multi-turn execution or final business outcomes. These observations justify targeted testing, not a timeless ranking of language difficulty.

#### International Tool Calling (ITC) Dataset

The [ITC dataset](https://arxiv.org/abs/2603.05515) introduces **region-specific APIs** — not just translated English APIs, but tools that actually exist in specific countries (e.g., local payment processors, regional weather services, domestic e-commerce platforms):

- **Scale:** 3,571 real APIs, 17,540 tasks, 20 categories, 40 countries
- **Split:** 15,790 training tasks, 1,750 test tasks (partitioned at API level to test generalization)
- **Key finding:** Fine-tuning on ITC dramatically improves non-English tool calling via better reasoning consistency and cross-lingual generalization

This adds a useful regional-tools dimension because real-world agentic systems need to call _local_ tools — not just English tools with translated interfaces.

---

## 3. Inference-Time Architecture Patterns

When you can't (or don't want to) retrain the model, these architecture patterns improve multilingual tool use at inference time.

### 3.1 The Translation Sandwich

A common architecture to evaluate:

```
User (Chinese) → Translate to English → LLM Reasoning + Tool Calls
→ Execute Tools → Translate Results → Respond in Chinese
```

**Pros:** Leverages the model's strongest (English) capabilities without retraining. Quick to implement. Works with any model.

**Cons:** Extra calls add deployment-dependent latency and cost. Translation may change entities, amounts, or domain meaning. Preserve exact identifiers and evaluate direct-language execution against the translated route; do not assume English pivoting always helps.

**When to use:** As a pragmatic first step for languages where you have no training data. Not a long-term solution for high-quality user experiences.

### 3.2 Multilingual Tool Descriptions

Optimizing instructions and descriptions is a useful engineering lever, but an English tool-efficiency result does not establish the multilingual benefit. Test localized examples against the same retrieval and execution baseline:

**Localized triggers** — Include `when_to_use` examples in target languages:

```json
{
  "name": "weather_forecast",
  "description": "Get weather forecast for a location",
  "when_to_use": "Weather, temperature, rain queries",
  "example_queries": [
    "What's the weather in Tokyo?",
    "東京の天気は？",
    "¿Cuál es el clima en Madrid?",
    "Какая погода в Москве?"
  ],
  "parameters": {
    "location": {
      "description": "City name or coordinates. Accepts names in any language.",
      "examples": ["Tokyo", "東京", "Москва", "São Paulo"]
    },
    "date_format": {
      "description": "Output date format. Auto-detected from user locale.",
      "enum": ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"]
    }
  }
}
```

This targets cross-language retrieval without retraining. Measure whether it actually improves selection and final task success; longer descriptions can also introduce noise.

### 3.3 MCP for i18n

The Model Context Protocol (MCP) is emerging as the standard for tool integration, and there are already i18n-specific patterns:

| Pattern                       | Description                                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **i18n MCP Servers**          | Dedicated tools for managing translation files (JSON locale files), querying, creating, and updating keys across locales |
| **Context-aware translation** | MCP connects LLMs to code repos + CMS for real-time context, reducing translation hallucinations                         |
| **Locale-aware routing**      | Route tool calls to region-specific API endpoints based on detected user locale                                          |
| **Missing key detection**     | Automated identification of untranslated strings across locale files                                                     |

The key advantage of MCP for i18n is that it standardizes how AI agents discover and interact with localization infrastructure — rather than each agent implementing its own ad-hoc translation layer.

### 3.4 Multi-Agent Architectures for Cultural Adaptation

Single-agent systems struggle with the dual optimization of linguistic accuracy and cultural adaptation. Multi-agent architectures decompose this into specialized roles:

#### Multi-Agent Translation Framework

A specialized orchestration for cultural-linguistic adaptation:

1. **Translation Agent** — Raw linguistic conversion, optimized for accuracy
2. **Interpretation Agent** — Embeds idioms, cultural references, and regional nuances
3. **Content Synthesis Agent** — Restructures content for target culture (e.g., different argumentation styles)
4. **Quality/Bias Evaluation Agent** — Catches cultural insensitivity and evaluates naturalness

This is more expensive than single-model inference but produces significantly better results for culturally sensitive applications — customer support, marketing content, legal communications.

#### Multi-Agent Debate for Cultural Alignment (ACL 2025)

Two LLM agents debate over a cultural scenario to collaboratively arrive at a culturally aligned decision. The key finding: this approach enables even **smaller LLMs to achieve performance comparable to much larger models** on cultural reasoning tasks, because the debate process surfaces implicit cultural assumptions that a single inference pass might miss.

#### CultureLLM (NeurIPS 2025)

Uses the World Value Survey as seed data, then applies semantic data augmentation to generate culture-specific fine-tuning data. The approach works even for low-resource languages because it generates training data from structured survey responses rather than requiring large corpora of culturally-specific text.

---

## 4. Failure Modes in Multilingual Tool Use

Understanding exactly _how_ tool use breaks in non-English contexts is essential for building robust systems. The following seven categories are a diagnostic checklist, not a measured ranking of production failure frequency:

### 4.1 Non-Latin Data Misinterpretation (Critical)

Test whether a model preserves CJK or Arabic names, numerals, and identifiers in tool results. Compare matched content in user messages and tool messages before attributing errors to the message role.

**Possible causes to distinguish:** unfamiliar entities, encoding or normalization changes, prompt-role effects, and task ambiguity. The evidence here does not isolate an attention-mechanism cause.

**Mitigation:** Post-process tool outputs to include both original and transliterated versions for critical data fields.

### 4.2 Internal Prompt Language Mismatch (High)

Agent frameworks (LangChain, CrewAI, etc.) often hardcode action-selection and reasoning prompts in English. When a Japanese user queries the agent, the user's message is Japanese, the tool descriptions are English, the internal reasoning prompts are English, and the model must context-switch between languages at every step.

**Mitigation:** Maintain language-specific prompt templates for the agent's internal reasoning chain. This is an underinvested area — most frameworks don't support this out of the box.

### 4.3 Cross-Lingual Semantic Collapse (Critical)

Direct translation of specialized terminology produces nonsensical output. Financial terms, medical jargon, and legal concepts often don't have 1:1 translations, and a generic translation model will produce terms that are linguistically valid but semantically wrong in the domain context.

**Mitigation:** Domain-specific glossaries injected into tool descriptions. Don't rely on the model to translate — provide explicit mappings for critical terms.

### 4.4 Negative Cross-Lingual Transfer (High)

Multilingual models trained on typologically distant languages (e.g., English + Mandarin + Finnish) can experience representation interference where one language's grammatical patterns corrupt another's.

**Mitigation:** MoE routing, language-specific LoRA adapters, or ensemble approaches where different models handle different language families.

### 4.5 Format Burden Amplification (Medium)

The difficulty of generating valid structured output (JSON, XML) compounds in non-English contexts. The model must simultaneously maintain JSON syntax, generate non-ASCII string content, and handle encodings — a harder constraint satisfaction problem than English-only structured output.

**Mitigation:** Grammar-based decoding (Outlines, vLLM grammar mode) to enforce JSON validity at the token level, independent of the language of string values.

### 4.6 Token Inflation (Medium)

Some localized text consumes more tokens than equivalent English; the ratio is tokenizer- and content-dependent. Log actual token counts and truncation events before treating context loss as the cause of a failure.

**Mitigation:** Language-aware context windowing, aggressive tool definition pruning for non-English sessions, or on-demand tool discovery (Anthropic's Tool Search Tool pattern).

### 4.7 Silent Evaluation Gaps (Critical)

Standard evaluation suites are English-centric. A model that passes all English tool-use benchmarks can still fail catastrophically in production for Korean or Arabic users — and these failures go undetected until deployment.

**Mitigation:** The MASSIVE-Agents and ITC benchmarks provide a starting point, but production systems need continuous multilingual evaluation with native-speaker review.

---

## 5. RL-Based Approaches: Closing the Loop

Reinforcement learning offers unique advantages for i18n improvement because many aspects of multilingual tool use are **verifiable** — you can check if the JSON parses, if the API call succeeded, and if the response is in the right language.

### 5.1 Extending RLVR to Multilingual Settings

Reinforcement Learning with Verifiable Rewards (RLVR) has proven powerful for math and code. The same framework extends naturally to multilingual tool use:

```python
def multilingual_tool_use_reward(
    prompt: str,
    response: str,
    expected_language: str,
    tool_schema: dict
) -> float:
    rewards = {}

    # 1. Format verification (language-agnostic)
    rewards["json_valid"] = 1.0 if parse_json(response) else 0.0

    # 2. API execution verification (binary, language-agnostic)
    rewards["api_success"] = 1.0 if execute_tool_call(response) else 0.0

    # 3. Response language match
    detected_lang = detect_language(extract_natural_text(response))
    rewards["language_match"] = 1.0 if detected_lang == expected_language else 0.0

    # 4. Cultural appropriateness (AI judge)
    rewards["cultural_score"] = cultural_judge(
        response, expected_language, prompt_context=prompt
    )

    # Weighted combination
    return (0.3 * rewards["json_valid"] +
            0.3 * rewards["api_success"] +
            0.2 * rewards["language_match"] +
            0.2 * rewards["cultural_score"])
```

This is illustrative pseudocode with arbitrary weights, not a validated reward design. Run tools in a controlled evaluation environment. A successful API response can still represent the wrong action; verify the final state against an oracle. Language detection and cultural judging are fallible proxies that need per-language validation.

### 5.2 Credit Assignment in Multilingual Pipelines

When a multilingual agentic task fails, **where did it go wrong?** The translation step? Tool selection? Parameter extraction? Output synthesis? This is the credit assignment problem, and two recent methods are particularly relevant:

**iStar (2025)** — Uses implicit step rewards derived from trajectory preferences. For multilingual tool use, this means training the model to identify _which step_ in the pipeline caused failure, rather than assigning blame to the entire trajectory. A trajectory where translation was perfect but tool selection failed should teach the model differently than one where translation was lossy from the start.

**MA-RLHF (Macro Actions, 2024)** — Treats "translate → call tool → translate back" as a single macro action for cleaner credit assignment. This is especially useful when the translation and tool-calling steps are tightly coupled — optimizing them independently leads to suboptimal results because the translation must anticipate the tool's parameter requirements.

### 5.3 Multilingual Reward Modeling

The reward model itself needs to be multilingual — or at least not monolingually biased:

| Approach                          | Pros                                                                           | Cons                                 |
| --------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------ |
| **Per-language reward models**    | Most accurate per language                                                     | Expensive to train and maintain      |
| **Cross-lingual reward transfer** | Efficient, uses semantic similarity to transfer reward signals                 | Lower accuracy for distant languages |
| **Multi-objective rewards**       | Balances language quality, cultural fit, and tool-use correctness (RLMR-style) | Complex hyperparameter tuning        |

**GRPO** (Group Relative Policy Optimization) is particularly useful here because you can form groups _per language_ — comparing French outputs against other French outputs rather than against English outputs. This reduces cross-language comparisons inside a group, but cannot remove bias in the reward function or guarantee culturally appropriate outputs.

---

## 6. Practical Playbook: What to Do Today

If you're building a multilingual agentic system, here's the priority order:

### Phase 1: Quick Wins (Days)

1. **Add multilingual examples to tool descriptions** — Include `example_queries` in your top 5 target languages. Evaluate retrieval and execution gains before rollout.
2. **Implement language detection on outputs** — Catch language-switching/mixing before responses reach users.
3. **Region-specific API routing** — Use the user's locale to select culturally appropriate endpoints (local weather services, payment providers, search engines).

### Phase 2: Architecture (Weeks)

4. **Language-specific prompt templates** — Don't just translate English prompts. Create unique templates that respect native terminology, sentence structure, and tone. Each language may need different tool selection phrasing.
5. **Grammar-based decoding for structured output** — Enforce JSON schema at the token level to prevent format corruption in non-English contexts (Outlines, vLLM).
6. **Build a translation sandwich with quality gates** — If using the translate→process→translate pattern, add verification checkpoints (back-translation consistency, term preservation checks).

### Phase 3: Training (Months)

7. **Fine-tune on ITC or MASSIVE-Agents data** — Keep training and evaluation splits separate, and supplement static calls with interactive tasks.
8. **Multilingual RLHF** — Collect preference data from native speakers in target languages. Choose the sample size from observed error coverage and learning curves; there is no universal 1,000-pair guarantee.
9. **Language-specific LoRA adapters** — Train lightweight adapters for each target language family, sharing the base model. Check cross-language regressions and adapter-routing cost; this does not automatically remove interference.

---

## 7. Emerging Trends (2025-2026)

| Trend                                   | Description                                                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sovereign AI**                        | Governments funding LLMs optimized for local languages, regulatory norms, and cultural datasets                                             |
| **Domain-specific multilingual models** | Healthcare and finance LLMs that work across languages within a domain, rather than general-purpose multilinguality                         |
| **Multilingual MCP ecosystem**          | Standardized i18n tool servers for translation management, locale detection, and cultural adaptation                                        |
| **Code-as-tool-orchestration**          | Using code execution (Python/JS) to orchestrate tool calls, reducing reliance on prompt-based tool selection that degrades across languages |
| **1M+ token contexts**                  | Maintaining full bilingual conversation history + tool schemas without truncation                                                           |
| **Multimodal multilingual agents**      | Image/video understanding aids cultural context — reading signage in local scripts, understanding regional UI patterns                      |

---

## 8. Key References

| Paper / Resource                                                                                                 | Venue            | Year | Focus                                                    |
| ---------------------------------------------------------------------------------------------------------------- | ---------------- | ---- | -------------------------------------------------------- |
| [Enhancing Function-Calling Capabilities in LLMs (Chen et al.)](https://aclanthology.org/2025.naacl-industry.9/) | NAACL            | 2025 | Multilingual function calling + translation pipeline     |
| [MASSIVE-Agents (Kulkarni et al.)](https://aclanthology.org/2025.findings-emnlp.1099/)                           | EMNLP            | 2025 | 52-language function calling benchmark                   |
| [International Tool Calling Dataset (Zhang & Zhu)](https://arxiv.org/abs/2603.05515)                             | arXiv / ICLR sub | 2026 | 3,571 real APIs across 40 countries                      |
| Bloomberg Context Optimization                                                                                   | ACL              | 2025 | Joint optimization of agent + tool descriptions          |
| CultureLLM                                                                                                       | NeurIPS          | 2025 | Culture-specific LLM fine-tuning from World Value Survey |
| Multi-Agent Debate for Cultural Alignment                                                                        | ACL              | 2025 | Collaborative cultural decision-making via debate        |
| CLCA (Cultural Learning-Based Adaptation)                                                                        | arXiv            | 2025 | Role-play-based cultural norm capture                    |
| Cross-Lingual Optimization (CLO)                                                                                 | various          | 2025 | Efficient English→Target language transfer               |
| TransLLM Framework                                                                                               | various          | 2025 | Translation CoT + LoRA + knowledge distillation          |
| iStar                                                                                                            | arXiv            | 2025 | Implicit step rewards for agentic credit assignment      |
| MA-RLHF (Macro Actions)                                                                                          | arXiv            | 2024 | Macro actions for long-horizon RL reward shaping         |

---

_Treat language as a deployment condition: measure the full workflow, inspect failures, and retain interventions only when they improve the outcome that matters._
