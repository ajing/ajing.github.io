---
author: Jing Lu
pubDatetime: 2026-07-03T18:00:00Z
title: "Scaling RL for White-Collar Work: The Environment Foundry"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Agents
  - Reinforcement Learning
description: "A practical framework for turning common white-collar workflows into RL environments: spreadsheets, CRM tasks, customer support, web research, dashboards, and other software-mediated work."
---

Reinforcement learning for language models has a supply problem.

Math and competitive programming scaled first because the reward is cheap: the answer is either correct or it is not. Software engineering came next because repositories contain tests, issues, commits, and runtime environments. But most economically valuable work is not a math problem or a coding benchmark. It is white-collar work done through software: updating spreadsheets, reconciling records, producing reports, operating CRMs, handling support tickets, researching vendors, cleaning data, preparing dashboards, checking policies, drafting documents, and moving state across tools.

The hard question is:

> How do we turn ordinary white-collar work into RL?

Not into demos. Not into prompt examples. Into trainable environments.

My current answer is that the next useful scaling unit is the **environment foundry**: a pipeline that turns open-source data, public task seeds, and synthetic business state into executable, verifiable agent environments.

The important word is **environment**. A job post, a GitHub issue, a forum question, a spreadsheet request, or a support policy is not yet RL data. It is a seed. It becomes RL data only after we wrap it in state, actions, observations, and a reward.

---

## 1. Why White-Collar Work Is Different From Math and Code

RL for LLMs works best when the reward is reliable. Math has exact answers. Code has tests. Many agent papers and systems lean on this property.

The software engineering line is especially instructive. [SWE-bench](https://www.swebench.com/) turns real GitHub issues into patching tasks. [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) packages real-world Python tasks with codebases, executable runtimes, unit tests, and natural-language task descriptions. [R2E-Gym](https://r2e-gym.github.io/) pushes further with procedurally curated executable software engineering environments. [SWE-RL](https://arxiv.org/abs/2502.18449) uses open software evolution data, including code snapshots, code changes, issues, and pull requests. [Agent-RLVR](https://arxiv.org/abs/2506.11425) studies why sparse RLVR struggles in agentic software settings and adds guidance to make environment rewards more useful.

The pattern is clear:

```text
open data -> executable state -> agent actions -> verifier -> reward
```

That is the template. The problem is that most white-collar work does not come with clean unit tests.

Consider a normal analyst request:

```text
Clean this messy vendor spreadsheet, standardize company names,
remove duplicates, classify each vendor by spend category, and
produce a short summary of the top cost-saving opportunities.
```

There are several different skills inside this one request:

- file understanding
- spreadsheet manipulation
- entity resolution
- classification
- numerical aggregation
- judgment about savings opportunities
- report writing

Some parts are deterministic. Some parts are fuzzy. Some parts require policy. Some parts need a human-quality rubric. If we score only the final written answer, the reward is too vague. If we score only row equality, we miss the actual business value.

White-collar RL needs hybrid environments: partly executable, partly policy-constrained, partly judged, and always stateful.

---

## 2. The Environment Is the Product

A useful RL environment for an LLM agent is not just a prompt. It is a packaged world:

| Component | Question it answers | White-collar example |
|---|---|---|
| Task distribution | What kind of work is sampled? | invoice reconciliation, CRM cleanup, data extraction |
| State/backend | What world does the agent operate on? | spreadsheets, SQLite DBs, browser pages, ticket queues |
| Action grammar | How can the agent act? | tool calls, SQL, browser clicks, sheet edits, document edits |
| Harness | Who executes actions and returns observations? | browser runner, Python sandbox, spreadsheet engine |
| Verifier/reward | How do we know whether the work succeeded? | cell checks, DB diffs, policy checks, rubric scores |
| Split discipline | How do we avoid memorization? | unseen templates, unseen websites, unseen company policies |

This is why environment infrastructure matters. Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) frames environments around datasets, harnesses, and reward functions. NVIDIA [NeMo Gym](https://docs.nvidia.com/nemo/gym/about) defines an environment as the complete system an agent interacts with, including dataset, harness, verifier, and state. OpenPipe [ART](https://github.com/OpenPipe/ART) and Microsoft [Agent Lightning](https://microsoft.github.io/agent-lightning/latest/) point in the same direction from the training side: RL systems need a way to collect trajectories from real multi-step agent execution.

There is also a separate systems layer for making those trajectories trainable at scale. [slime](https://github.com/THUDM/slime), from THUDM, is a useful reference point here: it is an LLM post-training framework for RL scaling that connects Megatron training with SGLang rollout, exposes custom data-generation and reward workflows, and keeps environment interaction, verifier feedback, rollout, and training in one explicit data path. In the environment-foundry framing, verifiers/NeMo Gym define what an environment is; slime is closer to the high-throughput factory floor that turns environment rollouts into model updates.

The bottleneck is not only PPO vs GRPO vs DPO. The bottleneck is whether we can manufacture enough reliable environments.

---

## 3. A White-Collar Work Taxonomy for RL

White-collar work is too broad to treat as one domain. The first step is to classify it by **state**, **action space**, and **verification path**.

Spreadsheets, slide decks, and documents are important, but they are only the artifact layer. They are where a lot of office work becomes visible. They do not cover the full job.

Most white-collar workflows combine five layers:

| Layer | Examples | Why it matters for RL |
|---|---|---|
| Office artifacts | spreadsheets, docs, slide decks, PDFs | concrete outputs, partial deterministic verification |
| Systems of record | CRM, ERP, ticketing, HRIS, finance systems | business state changes, permissions, no-collateral-damage checks |
| Communication | email, chat, meetings, customer conversations | multi-turn interaction, missing information, social constraints |
| Research and browsing | vendor pages, policies, public web, internal wiki | source grounding, extraction, contradiction handling |
| Judgment and policy | approvals, risk rules, prioritization, escalation | reward is partly rule-based and partly preference-based |

So the goal is not to say "Excel + PowerPoint = white-collar work." The goal is to use office artifacts as the easiest entry point into a larger environment distribution. A spreadsheet task often touches a CRM export, a manager's email, a metric definition, and a final deck. A support task may end in a note, but the real work is changing state in a policy-constrained system.

Here is a practical taxonomy:

| Work type | State | Actions | Verifier |
|---|---|---|---|
| Spreadsheet cleanup | `.xlsx` workbook, CSVs | formulas, Python, sheet edits | cell equality, schema checks, aggregate checks |
| Data extraction | websites, PDFs, docs | browser, OCR, parsing, CSV writing | field accuracy, source coverage, citation checks |
| BI/dashboard work | database, metric spec | SQL, Python, chart/report generation | metric equality, visual presence, rubric |
| Customer support ops | user, order, policy, tools | conversation plus API calls | final DB state plus policy compliance |
| CRM/admin cleanup | accounts, contacts, notes | CRUD tools, dedupe, classification | DB diff, no-collateral-damage checks |
| Procurement/vendor research | web, vendor docs, scoring rubric | search, extract, compare, summarize | citation accuracy, table completeness, rubric |
| E-commerce operations | catalog, inventory, orders | update records, generate copy, check stock | state diff, constraint checks |
| Document workflows | contracts, memos, templates | edit docs, redline, summarize | required clauses, formatting, citation checks |
| Marketing/content ops | brief, brand guide, CMS stub | draft, revise, schedule | rule checks plus judge/human preference |

The right abstraction is not "can the model answer the question?" It is "can the model move the environment from an initial state to an acceptable final state without violating constraints?"

That sounds small, but it changes the entire data strategy.

---

## 4. Freelancer Marketplaces Are Maps, Not Training Sets

Freelance marketplaces are useful because they reveal the distribution of small, paid white-collar tasks. Public category pages from Upwork, Fiverr, and Freelancer show recurring demand in data analysis, Excel work, web scraping, automation, data entry, dashboarding, AI services, writing, admin support, finance, and marketing.

But the safe lesson is:

> Use freelancer data as a task taxonomy and environment seed source, not as a pile of text to scrape into training.

There are three reasons.

First, job postings are often underspecified. A client says "build dashboard" but the real task depends on private data, business context, and follow-up negotiation.

Second, licensing and terms matter. Public visibility does not automatically mean the data is appropriate for model training.

Third, the job post is not the work. The work includes files, accounts, policies, messy edge cases, feedback, and final acceptance criteria.

The better pipeline is:

```text
public task signal
  -> abstract workflow type
  -> synthetic or licensed state
  -> executable tools
  -> verifier
  -> train/dev/test environment splits
```

For example, an Upwork-style "web scraper for product listings" request should not become "train on this job description." It should become a family of environments:

- local static e-commerce sites with different DOM structures
- target CSV schemas
- hidden gold tables
- allowed/disallowed domains
- penalties for fabricated rows
- train/test split by website template

Now it is RL fuel.

---

## 5. The Environment Compiler

I think the key missing system is an **environment compiler**.

It takes a work seed:

```text
"Need someone to clean a large Excel file, deduplicate companies,
normalize categories, and produce a summary dashboard."
```

And emits an environment specification:

```yaml
task_id: vendor_spend_cleanup_047
instruction: >
  Clean the vendor spend workbook, deduplicate vendor records,
  normalize categories, and create a summary table by category.
state:
  files:
    - input/vendor_spend_messy.xlsx
  hidden_gold:
    - gold/vendor_spend_clean.csv
    - gold/category_summary.csv
tools:
  - python
  - spreadsheet_editor
  - filesystem
actions:
  - inspect_workbook
  - edit_sheet
  - run_python
  - write_report
reward:
  deterministic:
    - schema_valid(output/vendor_spend_clean.csv)
    - duplicate_rate <= 0.01
    - category_accuracy >= 0.95
    - aggregate_mape <= 0.02
  rubric:
    - summary_mentions_top_three_categories
    - summary_flags_uncertain_vendor_matches
penalties:
  - deletes_required_rows
  - fabricates_vendor_names
  - overwrites_original_file
split:
  train: synthetic vendors from templates A-C
  dev: template D
  test: unseen templates E-F
```

This is the transformation that matters. The original request was natural language. The compiled environment is executable.

The same compiler pattern works across white-collar domains:

```text
support policy -> simulated user + tools + DB + policy verifier
spreadsheet forum post -> workbook + expected formulas + cell checks
dashboard request -> database + metric definitions + chart verifier
web research task -> browser snapshot + extraction schema + citation checker
CRM cleanup request -> synthetic CRM records + dedupe verifier
document review -> docx state + clause checklist + redline checks
```

The environment compiler is where domain knowledge enters the RL pipeline.

---

## 6. Three High-Leverage Prototype Environments

If I were building this from scratch, I would not start with every office workflow. I would start with three environment families where verification is strong enough to support RL.

### 6.1 Web Research to Structured Table

This is the freelance "find me X and put it in a spreadsheet" task.

Example:

```text
Find 50 suppliers of lab consumables that ship to California.
For each supplier, extract name, website, product category,
minimum order constraint, and source URL.
```

Environment:

- browser or static web snapshots
- target schema
- hidden gold set
- citation requirements
- anti-fabrication checks

Actions:

- search
- open pages
- extract fields
- write CSV
- cite source URLs

Reward:

- row recall
- field precision
- citation validity
- schema validity
- penalty for fabricated suppliers

This is not fully solved by static QA. The agent must decide where to search, when a source is enough, how to reconcile conflicting pages, and when to stop.

### 6.2 Spreadsheet Operations

This is the analyst automation task. It is one of the best white-collar RL candidates because the state is concrete and many outcomes are checkable.

Existing benchmarks already point this way. [SpreadsheetBench](https://spreadsheetbench.github.io/) builds spreadsheet manipulation tasks from real-world Excel forum questions. Recent [Spreadsheet-RL](https://arxiv.org/html/2605.22642v1) work explicitly targets RL-trained open-source spreadsheet agents. [DS-1000](https://ds1000-code-gen.github.io/) is also relevant because it turns practical data-science coding questions into reliable tests.

Environment:

- messy workbook
- task instruction
- formula constraints
- hidden answer workbook or derived checks

Actions:

- inspect sheets
- write formulas
- run Python
- create pivot tables or summary tabs
- export result

Reward:

- cell equality where exact
- aggregate equality where robust
- formula presence where necessary
- chart/table existence
- no unintended edits to protected ranges

This domain lets us train agents on the actual mechanics of office productivity, not just textual answers about office productivity.

### 6.3 Policy-Constrained Customer Operations

This is the "support agent with tools" task. The original [tau-bench](https://arxiv.org/abs/2406.12045) paper established the pattern; the current benchmark lineage has moved through tau2 into [tau3-bench](https://github.com/sierra-research/tau2-bench), which adds new domains and modalities while keeping the core idea: a domain has policies, tools, tasks, and a user simulator.

Example:

```text
A customer wants to return a delayed order, but the item is outside
the standard return window. The agent must authenticate the user,
check exception policy, choose whether to issue credit, update the
order state, and explain the decision.
```

Environment:

- customer profile
- order database
- policy document
- API tools
- simulated user

Actions:

- ask questions
- authenticate
- call tools
- update order/refund state
- respond to user

Reward:

- final database state
- policy compliance
- required authentication completed
- no unauthorized refund
- user-facing explanation quality

This matters because many white-collar jobs are not just "produce artifact." They are "act inside a business process without breaking policy."

---

## 7. Verification Is a Spectrum

The central mistake is to demand one reward type for every task. White-collar work needs layered verification.

| Layer | Example | Reliability |
|---|---|---|
| Exact state check | DB row updated, CSV schema valid | high |
| Numerical tolerance | revenue total within 0.5 percent | high |
| Programmatic invariant | no duplicate active accounts | high |
| Source-grounding check | every claim has a cited URL | medium |
| Policy automaton | refund allowed only under conditions | medium-high |
| LLM rubric | summary is clear and actionable | medium |
| Human preference | report is useful to a manager | high value, expensive |

The goal is not to eliminate judge models or humans. The goal is to reserve them for the parts that cannot be checked by code.

A good white-collar environment should maximize deterministic verification first:

```text
Can I check the final state?
Can I check the schema?
Can I check invariants?
Can I check numeric outputs?
Can I check citations?
Can I check policy preconditions?
```

Only after those checks should we ask a judge model whether the final narrative is good.

This is also how we reduce reward hacking. If the agent can get a high score by writing a confident summary while silently corrupting the spreadsheet, the environment is broken. The state verifier must fire before the prose rubric.

---

## 8. Why Current Benchmarks Are Necessary but Not Sufficient

The existing benchmark ecosystem already contains pieces of the answer.

[BrowserGym](https://github.com/servicenow/browsergym) gives a common framework for web agents. [WorkArena](https://servicenow.github.io/WorkArena/) moves browser agents into enterprise-style ServiceNow workflows. [OSWorld](https://os-world.github.io/) uses real computer environments for multimodal agents. These are closer to white-collar work than pure coding benchmarks because they involve UI state, workflows, files, and applications.

There are also more office-native efforts. [OfficeBench](https://github.com/zlwang-cs/OfficeBench) evaluates agents on multi-application office automation with synthesized documents, emails, and calendar events. [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) simulates consequential digital-worker tasks involving web browsing, code/program execution, and coworker communication. Spreadsheet benchmarks are getting more workflow-like: [SpreadsheetBench 2](https://spreadsheetbench.github.io/) focuses on end-to-end business spreadsheet workflows rather than isolated manipulations. Presentation work is starting to get its own benchmarks as well: [PPTC](https://github.com/gydpku/PPTC), [PPTArena](https://openreview.net/forum?id=Dl1S4EvFwh), and recent PowerPoint task-completion benchmarks test slide creation and in-place editing.

This makes the landscape less empty than it first looks. But it is still much thinner than software engineering. Many of these resources are evaluation benchmarks, not full RL-training environment foundries. They often have limited task counts, limited workflow coverage, or partial automation of rewards. More importantly, they cover visible artifacts better than they cover the hidden business systems that make white-collar work consequential.

But general office work still has missing pieces:

- more realistic private-business state without exposing real private data
- controllable synthetic companies, customers, vendors, and policies
- repeatable task generators rather than one-off benchmark instances
- reward functions that score both final state and process constraints
- train/test splits that prevent template memorization
- environment packaging that supports RL rollout collection, not only evaluation

This is why I prefer thinking in terms of an environment foundry rather than a benchmark suite.

A benchmark asks: "How good is the model?"

An environment foundry asks: "Can we manufacture more worlds where the model can practice economically meaningful work?"

---

## 9. The Training Recipe

Once the environments exist, the training recipe is not mysterious.

1. Define the action grammar first.
2. Build environments with reliable verifiers.
3. Collect successful trajectories from strong agents, humans, or search.
4. Supervised fine-tune on clean trajectories.
5. Add hard negatives and preference pairs.
6. Run RL only where the reward is stable enough.
7. Mine failures and expand the environment distribution.

This mirrors what has worked in agent post-training more broadly. The model should be trained against the same scaffold it will use at inference time. If production uses browser actions, train browser actions. If production uses spreadsheet tools, train spreadsheet tools. If production uses policy-constrained APIs, train with the same policy and API grammar.

At small scale, this can be done with a simple Python harness and a GRPO trainer. At larger scale, the systems problem becomes the main problem: rollouts dominate cost, model weights need to move between training and inference workers, long-horizon generations create tail latency, and custom environment code must feed rewards back without breaking the training loop. This is where a framework like [slime](https://thudm.github.io/slime/) becomes important. Its Megatron + SGLang design is opinionated, but that is the point: it optimizes for the hot path of RL scaling rather than abstracting every possible backend. For white-collar environments, the custom data-generation interface is the bridge from "run this spreadsheet/browser/support environment" to "produce rollouts, rewards, and training batches."

The common failure mode is scaffold mismatch:

```text
train: clean text tool-call examples
deploy: messy browser, flaky tools, ambiguous records, policy constraints
```

Then people blame the model. Often the environment distribution was the problem.

---

## 10. Scaling Law: More Environments, Not More Prompts

The naive way to scale white-collar agent data is to collect more prompts:

```text
10,000 spreadsheet prompts
10,000 research prompts
10,000 support prompts
```

That helps SFT, but it is not enough for RL. RL needs interaction, state, and reward.

The stronger scaling axis is:

```text
number of distinct executable environments
  x diversity of state distributions
  x verifier coverage
  x trajectory attempts per environment
```

For example, "web scraping" should not be one task. It should be a generator:

```text
site layout: table / cards / infinite scroll / PDF catalog
schema: simple / nested / normalized
noise: missing values / duplicates / misleading labels
constraints: rate limit / domain allowlist / citation required
reward: row recall / field accuracy / no fabrication
```

Now the model can learn the skill rather than memorize a site.

The same applies to spreadsheets:

```text
workbook shape: one tab / many tabs / merged cells / hidden sheets
operation: clean / join / aggregate / forecast / visualize
noise: typos / duplicate vendors / inconsistent dates
reward: exact cells / aggregate checks / protected ranges
```

And to customer support:

```text
domain: retail / airline / telecom / healthcare admin
policy: simple / exceptions / conflicting conditions
user behavior: cooperative / confused / adversarial / missing info
tools: lookup / update / refund / escalate
reward: final state / policy compliance / conversation quality
```

This is the real scale story.

---

## 11. What Open-Source Data Can Actually Provide

Open-source and public data can provide four ingredients:

### Task Seeds

Sources:

- GitHub issues and pull requests
- StackOverflow and Excel/forum questions
- public benchmark tasks
- public job categories and freelance listings
- public support policies and docs
- open datasets and Kaggle notebooks

Use: identify recurring work patterns.

### State Templates

Sources:

- open-source repos
- public CSV datasets
- synthetic companies/customers/vendors
- generated spreadsheets
- local static websites
- mock CRMs and ticket systems

Use: build executable worlds.

### Tool Grammars

Sources:

- APIs from open-source apps
- browser automation actions
- spreadsheet libraries
- SQL engines
- document tooling
- file-system sandboxes

Use: define what the agent is allowed to do.

### Verifier Patterns

Sources:

- unit tests
- schema validators
- DB diffs
- spreadsheet equality checks
- policy automata
- citation validators
- human or model rubrics

Use: turn outcomes into reward.

The data is not one monolithic corpus. It is an ingredient supply chain.

---

## 12. A Concrete First Build

If I wanted a credible v0 of "RL environments for white-collar work," I would build this:

### Environment 1: Data Extraction Agent

- 100 synthetic local websites
- 10 schema families
- hidden gold CSVs
- browser plus Python tools
- reward: field accuracy, citation validity, no fabrication

### Environment 2: Spreadsheet Agent

- 500 generated workbooks
- tasks from Excel forum taxonomies
- operations: clean, dedupe, join, aggregate, chart
- reward: workbook diff, aggregate checks, protected-range checks

### Environment 3: Support Ops Agent

- tau3-bench style simulator
- synthetic retail/telecom/banking/vendor-support domains
- SQLite backend
- policy documents
- reward: final DB state, policy compliance, conversation constraints

### Environment 4: Dashboard Analyst Agent

- public/synthetic business datasets
- metric definitions
- SQL/Python/report tools
- reward: metric equality, chart presence, written caveat rubric

That would be enough to test the thesis. Not "can the model answer white-collar questions?" but:

> Can an RL-trained open model become better at operating common office environments?

---

## 13. The Open Problems

There are real obstacles.

### Leakage

If the task comes from public web data, the model may have seen it. Train/test splits need to separate by template, website, policy, company, and generated state, not only by row.

### Verifier brittleness

Bad rewards teach bad behavior. A spreadsheet environment that accepts the right total while ignoring row corruption is not good enough.

### Long-horizon credit assignment

Many workflows have sparse final rewards. Agent-RLVR-style guidance, step-level checks, and intermediate state rewards may be necessary.

### UI instability

Browser and desktop environments are flaky. Static snapshots, mock apps, and deterministic local services are less glamorous but more trainable.

### Licensing and privacy

Freelance tasks, business docs, customer tickets, and real spreadsheets can be sensitive. A serious environment foundry needs licensed data, synthetic reconstruction, PII removal, and clear provenance.

### Human preference remains necessary

Some outputs are only useful if a human manager would actually trust them. The trick is to make human preference the top layer, not the whole reward.

---

## 14. The Takeaway

The path from open-source data to RL for white-collar work is not:

```text
scrape job posts -> train model
```

It is:

```text
find work patterns
  -> build stateful environments
  -> define action grammars
  -> write verifiers
  -> collect trajectories
  -> train and evaluate agents
  -> mine failures
  -> expand the environment distribution
```

Coding agents are ahead because code gave us tests. The broader white-collar world will need its own equivalent: spreadsheets with checkable outputs, CRMs with state diffs, support policies with compliance checks, research tasks with citation validators, dashboards with metric tests, and documents with structural requirements.

The labs that scale this will not just have better RL algorithms. They will have better environment foundries.
