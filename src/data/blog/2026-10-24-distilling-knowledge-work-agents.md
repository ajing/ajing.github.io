---
author: Jing Lu
pubDatetime: 2026-10-24T22:42:59-07:00
title: "What Makes a Good Teacher for Knowledge Work Agents"
featured: false
draft: false
tags:
  - AI
  - LLM Agents
  - Post Training
  - Evaluation
  - ML Engineering
description: "An overview of agent SFT distillation for research, spreadsheets, documents, and enterprise workflows, with a focus on teacher quality, student compatibility, and completed work."
---

Consider a normal analyst request:

> Compare the new supplier quotes with last quarter's purchases. Update the
> workbook, explain the price changes, and prepare a short memo with sources.

A strong agent completes the task. It locates the files, resolves supplier
names, checks currencies and tax treatment, updates the spreadsheet, and writes
the memo. We save its trajectory and use supervised fine-tuning, or SFT, to
teach a smaller model to do similar work.

The student learns the tool-call format. It produces a plausible sequence of
steps. But on a new task, it might compare quotes from different periods,
overwrite an existing formula, or write a conclusion that the source does not
support. These are possible failures in this illustrative workflow, rather
than results from a new experiment.

The teacher solved the original task. Whether its demonstration teaches the
student a reusable way of working is a separate question.

My reading of the literature is that **teacher quality depends on the student,
the work environment, and the supervision we extract**. A good teacher needs
to produce reliable decisions that the student can learn to execute, including
how to obtain evidence, use tools, recover from errors, and check the result.
Its own task score is useful, but does not determine all of those properties.

This overview focuses on knowledge work: research, documents, spreadsheets,
enterprise applications, and workflows that connect them. It draws on sources
available through **October 24, 2026**. Experimental findings are attributed to
their authors; I have not independently reproduced the training runs. Math and
software-engineering studies help explain mechanisms, with their task limits
kept alongside the claims.

## Table of contents

## Start with the work the student must deliver

Knowledge work usually combines several kinds of state. The supplier-analysis
task has documents containing evidence, a workbook containing calculations, and
a memo communicating the result. A customer-support task adds a conversation,
business rules, and records that may need to change. An internal research task
may connect a wiki, a database, and a presentation.

The student needs to preserve relationships across these objects. A number in
the memo should agree with the workbook. A workbook calculation should use the
intended records. A record update should follow the user's request and the
applicable business rules.

I find it useful to separate four parts of the capability:

| Part of the work      | What the agent must do                                   | What can go wrong                                           |
| --------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| Establish evidence    | Find, read, and reconcile relevant information           | Use an old quote or confuse two suppliers                   |
| Organize the workflow | Order dependent steps and move information between tools | Draft conclusions before checking the data                  |
| Execute operations    | Query systems and modify the intended objects            | Edit the wrong range or submit an invalid tool call         |
| Verify and deliver    | Inspect the result and satisfy the completion conditions | Save a file with broken formulas or omit the requested memo |

These are analytical categories, not independent modules that every agent must
implement. A single action can serve several purposes. Reading a worksheet can
establish evidence, reveal its structure, and change the next step of the plan.

Several benchmarks make different parts of this work observable.
[WorkArena](https://arxiv.org/abs/2403.07718) studies common operations in
enterprise software.
[OfficeBench](https://arxiv.org/abs/2407.19056) examines workflows across
applications.
[TheAgentCompany](https://arxiv.org/abs/2412.14161) places tasks in a simulated
company with files, internal websites, and coworkers.

These environments define useful evaluation problems. Their existence does
not establish that a particular distillation method works. They tell us what
sort of completed work a training claim should eventually explain.

This connects to my earlier notes on
[environments for white-collar work](/posts/2026-07-03-scaling-rl-for-white-collar-work-environments/).
The environment supplies the state, tools, and completion conditions. Teacher
selection determines which behavior we ask the student to learn inside that
environment.

## What gets distilled

In agent SFT, a training example can contain much more than a question and an
answer. It can include the task, previous observations, earlier actions, a
decision explanation, and the next tool call. The student learns to predict
selected outputs given that history.

A simple version of the pipeline is:

```text
task + initial files and application state
                  ↓
teacher executes a workflow
                  ↓
collect, filter, and format demonstrations
                  ↓
supervise selected student outputs
                  ↓
student executes new tasks independently
                  ↓
inspect output files, application state, and task completion
```

The collection policy and the training objective are separate choices. A
student can generate the history while an expert supplies the next action,
which is then learned through ordinary cross-entropy. A teacher trained with
reinforcement learning can produce fixed demonstrations for a student whose
update is entirely SFT.

Likewise, hard-label SFT and token-distribution distillation expose different
information. The former needs teacher outputs. The latter needs usable
teacher probabilities and a compatible representation. An output-only teacher
can still provide useful demonstrations and corrections.

There is also an important distinction between **using agents to create data**
and **teaching a model to act as an agent**. Synthetic reading-comprehension
answers can improve a document model without teaching it to navigate an
application, update a workbook, or complete a workflow.

## How existing work turns tasks into supervision

The literature offers several routes from work to training data. They differ
in what the student is asked to reproduce and how directly the evaluation
measures execution.

### Generate skill data from source material

[AgentInstruct](https://arxiv.org/abs/2407.03502) uses agentic flows to create
prompts and responses from raw sources such as documents and code. Its data
generation covers skills including reading comprehension, text modification,
and tool use.

Its relevance to knowledge work is the construction process: source material
can seed varied tasks, rather than requiring a manually written instruction
for every example. The student still learns the behaviors represented in the
resulting data. Improvements on those skills should be distinguished from
evidence of complete cross-application work.

### Make workflow structure explicit

[WorkflowLLM](https://arxiv.org/abs/2411.05451) collects workflows from Apple
Shortcuts and RoutineHub, converts them to a Python-style representation, and
expands the collection with generated queries and workflows. The supervision
contains plans, API operations, and control flow.

This makes dependencies teachable: obtain a record, inspect a value, choose a
branch, and pass the result to another operation. Its evaluation uses
reference-code metrics and model-based assessment. The reported Pass Rate
therefore should not be interpreted as a universal execution success rate in
live business applications.

### Learn interaction from executed trajectories

[WebDancer](https://arxiv.org/abs/2505.22648) constructs information-seeking
tasks, samples and filters trajectories, uses SFT for initialization, and then
applies RL. Search results and page observations become part of the context
for subsequent decisions. It provides a concrete example of teaching an agent
how to continue gathering information.

The training stages matter when reading its results. Some reported models use
SFT, while others also receive RL. Its information-seeking benchmarks cover
part of research work, rather than the entire quality of a long analyst report.

For computer interaction,
[Fara-7B](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/11/Fara-7B-An-Efficient-Agentic-Model-for-Computer-Use.pdf)
uses a multi-agent system to propose tasks, attempt solutions, and verify
trajectories. It extracts screenshots, reasoning text, and actions to train a
single computer-use model with cross-entropy SFT. This is a direct example of
transferring behavior from a more elaborate teaching system into a simpler
deployed model.

Taken together, these routes cover different supervision units: a skill
example, a workflow program, or a decision within an executed trajectory.
Treating them as interchangeable can hide what the student actually learned.

## The teacher is a system configuration

A model name identifies only part of the teacher. Its demonstrations also
depend on the tools, prompts, available information, planning procedure,
memory, retry budget, and verifier.

The Fara pipeline makes this concrete. A planning agent directs a browser
agent; selected browser-agent outputs become training targets. The student
learns from those outputs rather than replaying every internal message between
the teaching agents.

For knowledge work, I would distinguish three responsibilities within this
system:

- **Task construction** determines the distribution of work represented in the data.
- **Task solving** produces candidate decisions and execution paths.
- **Verification** decides which outputs satisfy the requested result.

A capable solver may produce narrow training coverage. A broad task generator
may create ambiguous or unsolvable requests. A verifier may accept an artifact
that looks complete while missing an important constraint. Improving one
responsibility does not automatically fix the others.

This also changes how to interpret generation cost. Failed attempts, repeated
verification, and environment execution all contribute to the cost of an
accepted demonstration. The cheapest individual teacher call need not produce
the cheapest useful training data.

## Task capability and teaching value can diverge

Direct teacher-selection evidence is stronger in reasoning distillation than
in broad knowledge work.
[Which Reasoning Trajectories Teach Students to Reason Better](https://aclanthology.org/2026.acl-long.1950/)
compares multiple teachers and students, finding that stronger teachers do not
consistently produce better students. Its Rank-Surprisal Ratio studies the
balance between informative supervision and compatibility with the learner.

[Local Naturalness](https://arxiv.org/abs/2510.03988) studies response selection
across teachers with long reasoning traces. It finds that global student
likelihood can be an unreliable selector and evaluates a local alternative.
Both papers concern mathematical reasoning; they provide mechanisms to
consider, rather than a ranking of teachers for office work.

In the supplier example, a teacher might solve everything through a compact
program that joins several files and applies complex matching rules. Another
valid path might inspect the schema, resolve ambiguous supplier names, and
calculate changes in stages. Which path is more useful depends on the
student's ability to reproduce and debug it. A shorter trace is not enough to
settle that comparison.

The distinction between correctness and learning value also appears in my
earlier discussion of
[SFT quality, novelty, and learnability](/posts/2026-09-27-sft-quality-novelty-learnability/).
A demonstration can meet the task specification while providing little new
capability, or while asking the student to learn too much at once.

[Reinforcement Learning Teachers of Test Time Scaling](https://arxiv.org/abs/2506.08388)
changes the teacher's objective directly. Given a question and a known
solution, the teacher learns to produce explanations rewarded through
student-related feedback. Its reward uses proxies for effective teaching; it
does not retrain a student on every candidate explanation to measure eventual
generalization. The work makes teaching quality an explicit optimization
target, with its main evidence in reasoning tasks.

For knowledge work, the corresponding principle is useful but conditional:
evaluate whether the teacher exposes a process that this student can absorb
and execute. Model size, fluency, and the teacher's own completion rate are
incomplete proxies for that outcome.

## Teach the states the student actually reaches

A complete successful demonstration shows how a task can proceed. During
deployment, the student may produce a different history.

It might open an older workbook, search for the wrong supplier name, or edit a
cell before understanding its formula. Those actions change the available
evidence and sometimes the underlying environment. The next teacher action
from the original demonstration may no longer be appropriate.

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses this general
imitation-learning problem by collecting expert supervision on states related
to the learner's behavior.
[GKD](https://arxiv.org/abs/2306.13649) brings teacher feedback onto
student-generated sequences for language-model distillation. The original GKD
experiments are language-generation tasks, rather than complete office
workflows, but the distinction between context source and target source is
important for agents.

[On-policy Expert Corrections](https://arxiv.org/abs/2512.14895) provides a
multi-turn agent example in software engineering. The student acts first;
the expert takes over partway through, and successful executions supply
supervision. It is partially on-policy and does not automatically inherit all
of the original DAgger guarantees.

[AgentRefine](https://arxiv.org/abs/2501.01702) studies trajectories containing
errors, environmental feedback, and corrections. It masks erroneous turns
while retaining context needed to learn recovery. Its ablations support
distinguishing an error in the history from an error used as a positive target.

This distinction is practical in knowledge work:

```text
student edits the wrong range       → retained as history
tool reports the resulting error    → retained as observation
teacher locates and repairs it      → eligible supervision
updated workbook is checked         → evidence about the repair
```

Preventing that mistake is a different training example. It starts before the
wrong edit, without the error observation. A recovery example starts after
the edit and must account for the changed state.

The same requirement applies when editing trajectories. Replacing an action
can invalidate later observations. A repaired action should be executed to
obtain a corresponding continuation, or used as a local target at the
appropriate prefix. Keeping an old successful ending after changing the
middle can create a demonstration that no agent could actually execute.

## Preserve the information behind the decision

Knowledge-work teachers often have information advantages: reference answers,
additional retrieval, richer interface structure, or instructions from a
planner. Whether that advantage transfers depends on what the student can
infer or obtain at deployment.

Fara offers a useful concrete distinction. Its data collection can use
accessibility-tree information, while the deployed student uses screenshots
and browser metadata. Training targets convert interface-element identifiers
into coordinates. Richer teacher information helps construct a target tied to
the student's visible interface.

Now suppose a teacher knows a supplier's payment terms from a document that
does not appear anywhere in the student's history. Directly imitating its
decision cannot teach the student how to find those terms. A reusable
demonstration would include obtaining the document or otherwise establishing
the relevant evidence.

[From Self-Distillation to Self-Practice](https://arxiv.org/abs/2609.29051)
reports failures from matching a privileged teacher view in multi-turn
AppWorld and SWE-bench settings. The student can inherit decisive behavior
without the information supporting it. This is a specific recent preprint,
rather than a universal result about every use of privileged information.

The useful boundary is whether the supervision can be grounded in the
student's observations and available information-gathering actions. A teacher
may use extra information to construct a better lesson. The lesson still
needs to show how the student can justify and carry out the decision.

## Keep the model compatible with its tools

The harness around an agent includes its system prompt, tool definitions,
execution hooks, and context management. These components influence which
strategies are practical for a particular model.

[Co-Evolving Harnesses and Models](https://arxiv.org/abs/2609.09134) reports a
counterexample in seven enterprise tasks. After adapting a harness to a weaker
model, full expert-trajectory imitation degrades that model's performance,
while local expert correction preserves more of the benefit. The authors
attribute the regression to a change in planning strategy that damages the
student's fit to the harness. The measured regression and that proposed
mechanism should be interpreted separately.

For the supplier task, a browser-only agent, a spreadsheet-API agent, and an
agent with a Python environment have different ways to complete the work.
Even two agents with identical tools may need different levels of planning
detail. The expert's preferred strategy can depend on abilities the student
has not acquired.

This is why teacher compatibility includes more than converting tool-call
syntax. A valid conversion preserves the meaning of observations and actions;
useful teaching also preserves a strategy the student can learn to execute.

## Evaluate the artifact and the resulting state

The final chat message provides limited evidence about completed work. The
supplier task also produces a workbook and a memo. A support task changes
records. A document-editing task should leave a usable saved file.

The evaluation literature gives several ways to inspect those results:

| Evaluation target              | Representative work                                    | What it adds                                                                          |
| ------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| Real spreadsheet requests      | [SpreadsheetBench](https://arxiv.org/abs/2406.14991)   | Multiple spreadsheet instances for an instruction, beyond a single worked example     |
| Business spreadsheet workflows | [SpreadsheetBench 2](https://arxiv.org/abs/2606.29955) | Generation, debugging, visualization, and separate local-edit and task-level measures |
| Cross-application work         | [OfficeBench](https://arxiv.org/abs/2407.19056)        | Information transfer and application switching within a workflow                      |
| Application state              | [AppWorld](https://arxiv.org/abs/2407.18901)           | Goal-state checks and checks for collateral changes                                   |
| Policy and user interaction    | [τ-bench](https://arxiv.org/abs/2406.12045)            | Business rules, dynamic conversations, and consistency across trials                  |
| Research reports               | [DeepResearch Bench](https://arxiv.org/abs/2506.11763) | Report quality and citation support assessed separately                               |

These are evaluation contributions, not demonstrations that one teacher is
best. They help define what a teaching result should measure.

SpreadsheetBench draws on real forum requests and associated files, including
structures more varied than simple relational tables. SpreadsheetBench 2
extends the workflow perspective to business workbooks. Its metrics distinguish
editing target cells correctly from satisfying the whole-workbook acceptance
conditions. The latter matters when a model fixes a requested value but
changes unrelated content.

AppWorld explicitly checks unexpected changes alongside goal completion,
allowing different valid paths through a task. τ-bench adds another important
dimension: the agent must interact with a user and follow domain rules, and
its behavior should remain reliable across repeated attempts.

For research, DeepResearch Bench separates report-quality assessment from
retrieval and citation assessment through RACE and FACT. These frameworks use
model judgments; they provide a structured evaluation rather than an
error-free certificate of truth.

The implication for teacher data is straightforward. A verifier that accepts
only a plausible final answer can miss errors that the student will imitate.
The acceptance conditions should reach the files, records, sources, and
constraints that constitute the work.

## Where SFT and RL meet

SFT provides explicit examples of behavior. It can establish action formats,
common workflows, and useful ways to respond to observations. Its coverage is
limited by the contexts and decisions represented in the demonstrations.

RL evaluates the student's own interaction through a reward. WebDancer
combines an SFT initialization with subsequent RL;
[DeepResearcher](https://arxiv.org/abs/2504.03160) studies RL with real web
search interactions. These are relevant routes for improving information
seeking, but their final performance should not all be credited to SFT
distillation.

For broader knowledge work, reward quality remains part of the problem.
Spreadsheet values and database changes can support executable checks.
Research depth, document usefulness, and presentation quality often require
rubrics or human judgment. A more sophisticated optimizer still depends on
what the evaluator rewards.

The literature therefore supports several combinations: demonstrations for
initial competence, expert labels on student-related states, and environment
feedback for further optimization. The appropriate combination depends on the
work and the available supervision. The evidence does not establish that every
knowledge-work agent needs the same sequence.

## What the literature says about a good teacher

The strongest general lesson is to evaluate a **teaching configuration**:
the task distribution, teacher system, student inputs, selected targets, and
acceptance conditions.

For knowledge work, I would organize the existing evidence around six
questions:

| Question                               | What useful supervision would contain                       | Current evidence boundary                                                            |
| -------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Does the workflow reach delivery?      | Decisions connecting the input materials to checked outputs | Individual studies cover different slices of work                                    |
| Are decisions supported by evidence?   | Information acquisition before conclusions or state changes | Privileged-view failures are documented in specific settings                         |
| Can this student absorb the strategy?  | Executable decisions at a suitable level of complexity      | Direct teacher-pairing evidence is stronger in reasoning tasks                       |
| Does it cover deviations?              | Recovery from states the learner can encounter              | Correction studies support this direction without establishing one best repair scope |
| Does it fit the deployment system?     | Compatible observations, tools, and planning behavior       | Enterprise harness studies provide conditional counterexamples                       |
| Is success checked at the right level? | Artifact, state, constraint, and source checks              | Every verifier still has a defined scope and possible blind spots                    |

These questions are a synthesis of the literature, rather than a validated
scoring formula. There is not yet sufficient evidence for a universal teacher
ranking across research, spreadsheets, documents, and enterprise operations.

For the analyst workflow at the start, the teaching target is a repeatable
process: establish which materials apply, make the intended changes, connect
the conclusions to evidence, and check the delivered artifacts. A teacher is
valuable when its demonstrations help the student perform that process on new
work under its own tools and information conditions.

## References

Years below follow the first public release of the linked work. Results belong
to the specified paper, task, and training setup; this list is not a common
leaderboard.

1. Alexandre Drouin et al. (2024). [WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?](https://arxiv.org/abs/2403.07718) arXiv:2403.07718.
2. Zilong Wang et al. (2024). [OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation](https://arxiv.org/abs/2407.19056) arXiv:2407.19056.
3. Zeyao Ma et al. (2024). [SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation](https://arxiv.org/abs/2406.14991) arXiv:2406.14991.
4. Harsh Trivedi et al. (2024). [AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://arxiv.org/abs/2407.18901) arXiv:2407.18901.
5. Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan (2024). [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) arXiv:2406.12045.
6. Frank F. Xu et al. (2024). [TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](https://arxiv.org/abs/2412.14161) arXiv:2412.14161.
7. Arindam Mitra et al. (2024). [AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/abs/2407.03502) arXiv:2407.03502.
8. Shengda Fan et al. (2024). [WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models](https://arxiv.org/abs/2411.05451) arXiv:2411.05451.
9. Jialong Wu et al. (2025). [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648) arXiv:2505.22648.
10. Ahmed Awadallah et al. (2025). [Fara-7B: An Efficient Agentic Model for Computer Use](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/11/Fara-7B-An-Efficient-Agentic-Model-for-Computer-Use.pdf) Microsoft technical report.
11. Rishabh Agarwal et al. (2023). [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) arXiv:2306.13649. GKD; accepted at ICLR 2024.
12. Stéphane Ross, Geoffrey Gordon, and Drew Bagnell (2011). [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) AISTATS, PMLR 15:627–635.
13. Niklas Lauffer et al. (2025). [Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections](https://arxiv.org/abs/2512.14895) arXiv:2512.14895.
14. Dayuan Fu et al. (2025). [AgentRefine: Enhancing Agent Generalization through Refinement Tuning](https://arxiv.org/abs/2501.01702) arXiv:2501.01702.
15. Zhou Yu et al. (2026). [Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up Where Imitation Fails](https://arxiv.org/abs/2609.09134) arXiv:2609.09134. Preprint.
16. Edoardo Cetin, Tianyu Zhao, and Yujin Tang (2025). [Reinforcement Learning Teachers of Test Time Scaling](https://arxiv.org/abs/2506.08388) arXiv:2506.08388.
17. Yuming Yang et al. (2026). [Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment](https://aclanthology.org/2026.acl-long.1950/) ACL 2026.
18. Hoang Anh Just, Myeongseob Ko, and Ruoxi Jia (2025). [Distilling Reasoning into Student LLMs: Local Naturalness for Selecting Teacher Data](https://arxiv.org/abs/2510.03988) arXiv:2510.03988.
19. Xingyu Su et al. (2026). [From Self-Distillation to Self-Practice: Privileged Information for Multi-Turn Agents](https://arxiv.org/abs/2609.29051) arXiv:2609.29051. Preprint.
20. Yuxiang Zheng et al. (2025). [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) arXiv:2504.03160.
21. Jian Zhu et al. (2026). [SpreadsheetBench 2: Evaluating Agents on End-to-End Business Spreadsheet Workflows](https://arxiv.org/abs/2606.29955) arXiv:2606.29955. Preprint.
22. Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao (2025). [DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](https://arxiv.org/abs/2506.11763) arXiv:2506.11763.
