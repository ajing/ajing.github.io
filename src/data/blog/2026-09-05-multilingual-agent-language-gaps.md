---
author: Jing Lu
pubDatetime: 2026-09-05T20:30:00-07:00
title: "Do LLM Agents Work Equally Well Across Languages?"
featured: true
draft: false
tags:
  - LLM Agents
  - Evaluation
  - Multilingual
  - AI
description: "Agent performance differs across languages, tasks, and localization settings. A visual review of 30 languages and 16 evidence sources identifies where Arabic, Thai, Japanese, and Hindi need targeted evaluation."
hideEditPost: true
visualReport: multilingual-agent-language-gaps
---

## Executive Summary

**No. Published evaluations show that the same LLM agent can perform materially differently across languages.** The practical conclusion is to evaluate **model × language × task × localization setting**. English performance alone does not establish reliability in another language. The evidence supports unequal performance; it does not support a permanent ranking of languages.

- **Arabic: repeated gaps in research and desktop tasks.** In audited, tool-assisted GAIA tasks, Arabic trails English by **19.4–30.3 percentage points** across three earlier flagships. In macOSWorld, both specialized computer-use agents score lower in Arabic when instructions and the operating-system interface switch together. These results concern generic Arabic labels; they do not establish Egyptian Arabic performance. [GAIA-v2-LILT](https://arxiv.org/html/2604.24929v1#S6), [macOSWorld v4](https://arxiv.org/html/2506.04135v4#S5).
- **Thai: the risk becomes clearer when the whole business workflow is localized.** Kimi K2.5 scores **56.1% in English, 57.3% with Thai dialogue only, and 32.7% with the full Thai retail workflow localized**. Policies, tools, and database content matter beyond conversational fluency. Earlier web-shopping results also flag Thai. [SEATauBench](https://arxiv.org/html/2606.28715v1#A6), [X-WebAgentBench](https://arxiv.org/html/2505.15372v1#S3).
- **Japanese and Hindi require narrower conclusions.** They are among Kimi K3's lower document-parsing observations, but Japanese desktop results change direction by agent, and Hindi GAIA scores rise substantially after benchmark auditing. Weakness depends on the task and evaluation design. [Kimi K3 results](https://huggingface.co/moonshotai/Kimi-K3/commit/26875c9de9f2fdd76741e66a9dd14e7a97b4fe2e).

**For the newest frontier models, the 30-language evidence is still incomplete.** Most controlled multilingual execution results come from earlier flagships. Current-model failures are useful deployment signals, but a failure observed in Russian, for example, does not prove that Russian caused it without an English control. This review covers 16 benchmark/resource projects and preserves counterexamples, missing evidence, and openness limits. Evidence cutoff: **September 5, 2026**; no new model evaluations were run.

## Auditing the benchmark narrows language gaps without eliminating them

**Arabic remains below each model's English reference after the GAIA audit.** The first chart shows English minus audited-language pass@1: higher bars mean a larger remaining gap. German, Hindi, Korean, and Brazilian Portuguese have smaller gaps, with different magnitudes by model.

[View figure: Remaining gap from each model’s English reference](#gaia_residual)

**Every audited language improves for every evaluated model.** The second chart shows audited minus minimally translated scores. Audits modify answer alignment, cultural context, and difficulty together: neither the entire improvement nor the entire residual is a pure language effect. [Table 2](https://arxiv.org/html/2604.24929v1#S6).

[View figure: Score improvement after task auditing](#gaia_lift)

Both charts use the same experiment, not independent replications. MAPS-GAIA and its audited version also share task ancestry. These are GPT-5.4, Gemini 3.1 Pro, and Opus 4.6 results, with 165 tasks per language in an actual tool-assisted workflow. They do not measure today's models directly.

## Conversational fluency does not establish local business competence

**Thai retail performance falls when localization extends beyond dialogue.** The chart compares an English baseline, localized dialogue, and fully localized business content for the same earlier flagship. Vietnamese full-localization retail performance slightly exceeds English, an important counterexample to a universal decline.

[View figure: Retail success by extent of localization](#sea_localization)

Thai fully localized airline and telecom results also fall below their English references; those observations remain in the chart data. The user simulator changes language too and can make errors, so the measured decline belongs to the overall system. It cannot all be assigned to the evaluated agent. Results use three trials per task; domains are not pooled. [SEATauBench Tables 9, 10, and 13](https://arxiv.org/html/2606.28715v1#A6).

## Desktop results repeat the Arabic signal and complicate the Japanese story

**Both specialized computer-use agents score lower in Arabic than in English.** Japanese and Russian move in opposite directions across agents: above English for OpenAI CUA, below English for Claude CUA. Compare each agent with its own English bar.

[View figure: Desktop task success by agent and language](#macos_language)

These 2025 model versions run 171 comparable tasks. Instructions and the OS interface change together, combining reading, layout, and planning effects. The authors document Arabic interface localization failures, including positioning errors. This motivates targeted interface testing, without establishing that right-to-left text alone causes the gap. [macOSWorld v4 Table 3 and cases](https://arxiv.org/html/2506.04135v4#S5).

## Translating the workflow into English can make performance worse

**Thai is the lowest original-language shopping score among the 11 target languages shown.** Translating to English helps French but hurts several others, particularly Arabic and Urdu. The y-values are WebShop task scores, not binary purchase-success percentages.

[View figure: Web shopping scores: original language versus English translation](#xweb_translation)

This earlier GPT-4o experiment adds a different task family to the Thai concern. It cannot be averaged with SEATau into a language risk score. Swahili's original-language result is comparatively strong, contradicting a simple rule that lower-resource languages always perform worst. The figure compares two published strategies, not every strategy in the paper. [X-WebAgentBench Table 2](https://arxiv.org/html/2505.15372v1#S3).

## Corroboration should include disagreement and missing controls

**More sources should constrain the conclusion as well as support it.** A paper, its GitHub repository, and its leaderboard form one evidence chain. An audited derivative is not a fresh independent task sample. The following table is a reading guide, not a ranking.

[View table: Language signals alongside counterevidence](#triangulation)

Amharic and Kannada warrant additional testing, with a narrower basis: in MASSIVE-Agents' 10k, zero-shot AST setting, Amharic is Nova Premier's lowest language and Kannada is Claude 3.5 v2 Sonnet's lowest. These are historical static function-call results. A later synthetic dataset does not independently replicate those performance findings. [MASSIVE-Agents Table 2](https://aclanthology.org/2025.findings-emnlp.1099.pdf).

## The broader evidence map distinguishes results from test resources

**Nine named benchmark columns cover 27 of the 30 target languages in some form.** A value of 2 means a verified per-language result in this broader historical/resource layer; 1 means task or aggregate coverage. Neither is an ability score. A dagger marks related Arabic or Filipino labels rather than an exact target-variety match.

[View figure: Broader evidence map · 1–15](#broad_atlas_1)

The first half combines static calls, execution tasks, and reusable resources. Dense coverage means more ways to investigate a language, not evidence that today's frontier models have passed them.

[View figure: Broader evidence map · 16–30](#broad_atlas_2)

Javanese has historical MASSIVE function-call evidence; Marathi has voice-tool resources; Gujarati has MAST retrieval coverage. Hausa's synthetic tasks do not establish current-model performance. Nigerian Pidgin has a separate [code-switching speech component resource](https://huggingface.co/datasets/mosesdaudu/switchboard-tierb-codeswitch), outside this agent-results grid. Egyptian Arabic and Western Punjabi still lack an included exact-match result.

## What has actually failed on current models?

The following evidence retains the review's current-model observations. It can identify execution failures and component weaknesses. Without a matched language control, it cannot identify how much of a failure is caused by the language.

## Valid formatting can hide substantial task errors

**Sol passes formatting on 99.9% of Russian tool-plan samples but passes the full sample on 69.1%.** Each bar splits all samples into three mutually exclusive parts. The middle segment shows errors that a JSON-format check would miss.

[View figure: All samples: format and content outcomes](#format_partition)

GorillaHard grades static outputs covering calls, abstention, and clarification; it does not execute tools. The documented dataset has 1,169 items, while submission-specific evaluated counts and repetitions are not separately disclosed. Model settings are not fully aligned, so this chart does not select a model winner. [Sol](https://mera.a-ai.ru/en/text/submits/2.0/8), [Opus 5](https://mera.a-ai.ru/en/text/submits/2.0/1), [Grok 4.6](https://mera.a-ai.ru/en/text/submits/2.0/11).

## Correct tool selection can still produce incorrect arguments

**About 8.1%–16.3% of call-required samples select matching tools without matching all arguments.** The middle segment points to concrete acceptance checks: entities, amounts, dates, and argument transfer between steps.

[View figure: Call-required items: tools and arguments](#argument_partition)

This denominator contains only call-required items, unlike the previous figure. The nested metrics share a denominator within this chart, enabling subtraction; the two figures cannot be joined into a funnel. No English comparison establishes a Russian penalty. [Scoring implementation](https://github.com/MERA-Evaluation/MERA/blob/0e1f4840baa313598266d5e63ac00a9f88a7b1df/benchmark_tasks/gorillahard/utils.py).

## Knowing when to stop needs its own acceptance test

**On abstention-required items, roughly 33.9%–41.7% lack the correct abstention.** False abstention on call-required items is much rarer. The chart separates these conditional error rates; the denominators appear in the category labels.

[View figure: Abstention errors under two conditions](#decision_errors)

These rates cannot be added, and a static output error is not an observed unauthorized real-world action. Abstention conditions follow task rules and are not all safety-policy refusals. [GorillaHard definition](https://github.com/MERA-Evaluation/MERA/blob/0e1f4840baa313598266d5e63ac00a9f88a7b1df/benchmark_tasks/gorillahard/README.md).

## Real code execution reveals reproducible task failures

RuBench asks agents to modify real repositories from Russian issue instructions, then runs regression tests. The chart separates **audit-retained passes, raw passes removed for answer exposure, and original failures**.

[View figure: Russian code tasks: audited run outcomes](#execution_partition)

Opus 5 fully passes **0/3 runs on each of two HTTP fixes**: method/header case rules and an empty-filename parsing exception. These are concrete task failures, without evidence that Russian caused them. Sol and Opus use different task sets and harnesses. Audit deductions re-score existing network-enabled runs; they are not offline reruns. [Pinned run record](https://github.com/eugeneshilow/rubench/blob/4b5c8da1b18ea85171b4270cda187ca667ab3c18/1.0/rounds/round-02/RESULTS.md).

## Four document languages deserve targeted end-to-end checks

**Thai, Japanese, generic Arabic, and Hindi sit at the lower end of Kimi K3's 13 included language observations.** The median line locates observations within this sample; it is not a deployment threshold.

[View figure: Document parsing across 13 language observations](#document_profile)

These scores combine text, layout, formula, and table parsing. **They are not agent success rates.** Language documents are not paired for identical content, and Arabic varieties are not separated. Validate the complete sequence: read a field, call the API, and inspect the resulting state. [Pinned author results](https://huggingface.co/moonshotai/Kimi-K3/commit/26875c9de9f2fdd76741e66a9dd14e7a97b4fe2e).

The script groups overlap substantially, and Korean scores **89.9**. A blanket non-Latin-script weakness label would conceal this counterexample.

[View figure: Parsing score distributions by script group](#script_distribution)

Boxes show the middle half of language observations; whiskers show minima and maxima. This summarizes 13 observations, not confidence intervals or the causal effect of a writing system. [Parsing metric definitions](https://arxiv.org/html/2603.28130v1#S3.SS4).

## Evidence for the current frontier is still sparse

**These two maps include only this review's September current-model cohort.** The broader historical results and resources above are excluded. A value of 1 marks included evidence, 0.5 marks the related Arabic label, and 0 means no included result was found.

[View figure: Current-model evidence map · 1–15](#atlas_1)

In the first half, Russian has code execution and static tool-plan results. French, Spanish, and other languages have document-component results, which do not establish browser or multi-turn business reliability.

[View figure: Current-model evidence map · 16–30](#atlas_2)

In the second half, Korean, Thai, and Italian have document-component results. Historical Javanese function-call evidence does not supply a current execution guarantee. Gurmukhi Punjabi cannot substitute for Western Punjabi. No current-model matched-language experiment was included in the paired-comparison column.

## Turn suspected mechanisms into verifiable tasks

Urdu and Iranian Persian offer concrete test conditions involving numerals, dates, and mixed-script identifiers. These are proposed fixtures, not measured current-agent failure rates. [Unicode number conventions](https://www.unicode.org/reports/tr35/tr35-numbers.html).

[View table: Proposed acceptance tests](#action_map)

**The most informative next experiment holds model, tools, budget, and business objective fixed, changes the language, and checks the final state.** Add locally authored tasks alongside those controls: matched tasks help estimate language gaps; native tasks test practical usefulness.

## Scope, metrics, and sources

This review checks published papers, author repositories, and dataset cards without running models. Its 16 core benchmark/resource entries are not an exhaustive census of public research. The fixed 30-language scope follows total-speaker coverage after excluding English and Chinese varieties, using a [pinned public transcription](https://en.wikipedia.org/w/index.php?title=List_of_languages_by_total_number_of_speakers&oldid=1370552712). Population determines inclusion, not training resources or ability.

The September 5 frontier scope includes Astra, Sol, Fable 5.1/5, Opus 5, Muse Spark 1.3, Grok 4.6, Kimi K3, GLM 5.3, and near-boundary Gemini 3.8 Flash/Qwen3.8. Per-language usable results exist for only a subset. Earlier flagships are labeled separately throughout. [Cohort reference](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-2).

[View table: Three current-model evidence layers](#methods)

Model-only aggregate scores from MCP Atlas and a tiny German evaluation without original traces do not fill per-language gaps here. “Not found” refers to this search and its inclusion rules. The lookup below keeps all 30 target languages and identifies useful evidence and next checks.

[View table: All 30 languages: evidence and next checks](#language_lookup)

## Public evidence has several different levels of openness

**A public paper, an open task set, evaluation code, and raw runs are different assets.** Terminal-Bench-LILT exposes samples but requires contacting the authors for the full set; the community LILTBench collection is separate. MASSIVE-Agents' converted dataset is publicly available, superseding an earlier “not released” characterization.

[View table: 16 projects: measurements and openness](#evidence_inventory)

VoiceAgentBench and TelcoAgent add task coverage, without current-frontier results. MAST currently provides a non-frontier baseline; its scheduled final competition results are still in the future at the review cutoff. These entries expand test resources, not the flagship-performance charts.

## What to test next

**Start with languages that have repeated task-specific signals, then fill commercially important blind spots.** Test Thai dialogue-only, tool, and full-business localization separately. Separate standard Arabic, dialects, and right-to-left interfaces. Split Japanese reading, visual targeting, and execution. Audit Hindi answer keys and regional rules before interpreting low scores. Prioritize other languages by local usage and error cost, not by how little research exists.

Fix the exact model, tool versions, task budget, and acceptance criteria. Retain every tool call and final state; for speech, retain the original audio and a correct-transcript control. The open question is which current-model failures persist after separating hearing, reading, tool selection, argument formation, and execution. None of these proposed follow-up tests has been run for this review.
