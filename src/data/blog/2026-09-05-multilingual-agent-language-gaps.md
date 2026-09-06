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
---

## Executive Summary

**No. Published evaluations show that the same LLM agent can perform materially differently across languages.** The practical conclusion is to evaluate **model × language × task × localization setting**. English performance alone does not establish reliability in another language. The evidence supports unequal performance; it does not support a permanent ranking of languages.

- **Arabic: repeated gaps in research and desktop tasks.** In audited, tool-assisted GAIA tasks, Arabic trails English by **19.4–30.3 percentage points** across three earlier flagships. In macOSWorld, both specialized computer-use agents score lower in Arabic when instructions and the operating-system interface switch together. These results concern generic Arabic labels; they do not establish Egyptian Arabic performance. [GAIA-v2-LILT](https://arxiv.org/html/2604.24929v1#S6), [macOSWorld v4](https://arxiv.org/html/2506.04135v4#S5).
- **Thai: the risk becomes clearer when the whole business workflow is localized.** Kimi K2.5 scores **56.1% in English, 57.3% with Thai dialogue only, and 32.7% with the full Thai retail workflow localized**. Policies, tools, and database content matter beyond conversational fluency. Earlier web-shopping results also flag Thai. [SEATauBench](https://arxiv.org/html/2606.28715v1#A6), [X-WebAgentBench](https://arxiv.org/html/2505.15372v1#S3).
- **Japanese and Hindi require narrower conclusions.** They are among Kimi K3's lower document-parsing observations, but Japanese desktop results change direction by agent, and Hindi GAIA scores rise substantially after benchmark auditing. Weakness depends on the task and evaluation design. [Kimi K3 results](https://huggingface.co/moonshotai/Kimi-K3/commit/26875c9de9f2fdd76741e66a9dd14e7a97b4fe2e).

**For the newest frontier models, the 30-language evidence is still incomplete.** Most controlled multilingual execution results come from earlier flagships. Current-model failures are useful deployment signals, but a failure observed in Russian, for example, does not prove that Russian caused it without an English control. This review covers 16 benchmark/resource projects and preserves counterexamples, missing evidence, and openness limits. Evidence cutoff: **September 5, 2026**; no new model evaluations were run.

## Explore the visual evidence

The complete English article below contains **15 interactive charts, 5 tables,
a 30-language lookup, and an audit of 16 benchmark and resource projects**.
Each chart retains its source, metric definition, and comparison limits.

**[Open the complete visual article in a full-width page →](/reports/multilingual-agent-language-gaps/)**

<iframe
  src="/reports/multilingual-agent-language-gaps/"
  title="Complete visual article: multilingual agent performance across 30 languages"
  width="100%"
  height="1050"
  style="border: 1px solid currentColor; border-radius: 12px; width: 100%; height: 85vh; min-height: 640px;"
  loading="eager"
></iframe>

The figures cover benchmark-audit sensitivity, full business localization,
computer use, translation strategies, tool arguments, abstention, code execution,
and document parsing. The broader evidence map and current-model map are kept
separate so historical results cannot be mistaken for current-model guarantees.

For reuse, [download the self-contained HTML article](/reports/multilingual-agent-language-gaps/index.html)
or inspect the [structured evidence snapshot](/reports/multilingual-agent-language-gaps/artifact.json).
