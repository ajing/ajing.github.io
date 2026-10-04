---
author: Jing Lu
pubDatetime: 2026-10-03T18:00:00-07:00
title: "How OSWorld Turns Computer Work into an Agent Benchmark"
featured: true
draft: false
tags:
  - LLM Agents
  - Evaluation
  - Reinforcement Learning
  - ML Engineering
description: "A technical guide to OSWorld's origins, 13 real tasks, Verified, 2.0 and 2.1, scoring contracts, and the experiments needed to distinguish local progress from complete computer work."
hideEditPost: true
---

An agent reads the receipts, finds the relevant emails, checks the bank records,
and recovers the employee's details from an old expense report. Hundreds of
actions later, the reimbursement request still has not been submitted.

This is a real failure pattern in OSWorld 2.0. The agent can perform useful
local work while failing to deliver the user's result. A higher-quality
screenshot model may help it find the next button. A larger context window may
help it retain a receipt. Neither improvement, by itself, establishes that it
can coordinate the entire workflow.

OSWorld makes this gap observable. Its original contribution was to turn
computer work into a reproducible experiment: **a user goal, an initialized
computer, and an executable definition of completion**. Its subsequent
evolution changes the reliability and scope of that experiment. Verified
repairs the measurement signal. Version 2.0 measures longer, connected work.
Version 2.1 pins the components needed to reproduce a run.

This post follows that evolution through the task-construction pipeline,
13 concrete tasks, evaluator implementations, and implications for long-horizon
reinforcement learning. My main conclusion is that **OSWorld 2.0 makes maintaining
and updating task state until delivery a central part of computer-use evaluation**.
Its results depend on the specified environment and tool protocol. The score does not isolate the underlying
model's intelligence, and a high partial score does not establish completion.

Sources were checked through **October 3, 2026**. Historical results below retain
their paper version, task release, and execution conditions. This is a source
review and research design; I did not run the full benchmark, access the gated
2.1 task implementations, or train a new agent. Explanations of what a task
diagnoses are my analysis unless explicitly attributed to its authors.

## Table of contents

## Why computer work needed a different benchmark

Consider a simpler request: a tuition-payment spreadsheet is available, and a
reminder email has already been drafted. Find the people who have not paid and
put their email addresses in the recipient field.

The agent must interpret the payment status, associate each unpaid record with
an address, and transfer the selected information to the email application.
The instruction specifies a result; it does not enumerate the clicks. An agent
can describe a correct plan and still select the wrong row, type into the email
body, or lose track of which people it has processed.

The [original OSWorld paper](https://arxiv.org/html/2404.07972v2) places this
problem in the earlier evaluation landscape. Static demonstrations lack a
running environment. Browser and mobile environments cover important
interactions but constrain applications and action spaces. OSWorld supplies a
computer on which browser tabs, desktop applications, files, and a terminal can
participate in the same task.

The result is a useful change in experimental unit:

```text
instruction + computer state
         → observation → action → changed state → observation → ...
         → output files / application state → evaluator
```

Alternative valid paths can produce the same accepted result. A menu action
and a keyboard shortcut need not be judged against one canonical click trace.
That flexibility makes the evaluator more important: accepting multiple paths
requires a precise definition of the result those paths must produce.

“Open-ended” describes the variety of work the environment can support. It
does not mean that the first task set represents every occupation, operating
system, or real user's workload.

## How the original task set was built

Tianbao Xie and collaborators released OSWorld in April 2024; the paper appeared
at NeurIPS 2024. The main benchmark contains **369 Ubuntu tasks**, with a separate
set of 43 Windows tasks used for cross-platform analysis. Platform support and
equal-sized task collections on every platform are different claims.
[Original paper, Sections 3.1–3.2](https://arxiv.org/html/2404.07972v2#S3),
[official repository](https://github.com/xlang-ai/OSWorld).

The authors describe nine annotators with computer-science backgrounds,
over three months of work, and roughly 1,800 person-hours. The work
included task collection, initial-state setup, evaluation code, and review.
Building a computer benchmark is therefore substantially more expensive than
writing its natural-language instructions.

The authors searched tutorials, application documentation, videos, forums,
courses, and personal blogs. Popularity, practical value, and diversity guided
selection. Cross-application workflows were harder to find in public materials,
so the authors also constructed them from everyday experience and combinations
of useful operations. Another 84 tasks were adapted from NL2Bash, Mind2Web,
SheetCopilot, PPTC, and GAIA.
[Collection details](https://arxiv.org/html/2404.07972v2#A2).

Ubuntu was an engineering choice. Open-source applications, accessible
interfaces, and permissive distribution conditions make it easier to rebuild
an environment and inspect its results. The paper's application-selection
criteria also include availability, popularity, active communities, learning
materials, and diversity. Commercial software and platform licensing complicate
equivalent public distribution on other operating systems.

This produces a selection effect worth carrying into interpretation. A useful
task enters the benchmark only if its starting state can be reconstructed and
its result can be checked. The collection consequently reflects both user needs
and experimental feasibility. It is not a random sample of production work.

## A task is an instruction, a state, and a verifier

A task specification needs three coordinated components:

| Component | What it establishes | Example |
| --- | --- | --- |
| Instruction | The requested result and constraints | Rename a sheet, make a backup, and preserve its contents |
| Initial state | Available evidence, files, and application state | A particular spreadsheet loaded in Calc |
| Evaluator | Evidence retrieval and acceptance rules | Inspect sheet names, order, and cell data |

The original JSON configurations expose fields such as `instruction`,
`config`, `source`, and `evaluator`. Setup operations prepare the workspace;
evaluation operations retrieve and compare the result. A useful abstraction,
rather than a literal configuration schema, is:

```python
task = {
    "goal": user_request,
    "initialization": [restore_base_vm, load_files, open_applications],
    "evaluation": [postprocess_if_needed, retrieve_evidence, check_result],
}
```

### Initial state is part of the problem

Computer assistance often starts midway through work. A document is already
open. The cursor is in a particular paragraph. An email draft exists. Some
browser tabs contain relevant evidence and others contain distractions.

OSWorld combines a base VM snapshot with task-specific initialization, avoiding
a separate multi-gigabyte snapshot for every task. The initialization must
produce the state the instruction assumes. If an application has not finished
loading when setup moves the cursor, an otherwise capable agent can receive the
wrong problem. That is an environment error, not evidence that it misunderstood
the user.
[Environment design](https://arxiv.org/html/2404.07972v2#S2.SS2).

![A task moves from a base VM through task initialization, agent interaction, postprocessing, evidence retrieval, and scoring.](/images/osworld/task-lifecycle.svg)

### Observation and action conditions define the capability being measured

The original environment supports screenshots, accessibility trees, or their
combination. An accessibility tree exposes structured interface information
such as element names, roles, states, and coordinates. A screenshot-only agent
has a different information channel.

Actions commonly use PyAutoGUI to express mouse and keyboard operations, plus
`WAIT`, `DONE`, and `FAIL`. One generated action can itself contain several
low-level operations. Terminal access, application scripts, direct APIs, and
extra tools can further change the problem. A score needs those conditions
alongside the model name.
[Observation and action spaces](https://arxiv.org/html/2404.07972v2#S2.SS3).

The 2024 paper reported 72.36% human performance and a best baseline of 12.24%.
That baseline used accessibility-tree observations and a 15-step budget; the
human participants were students with basic software familiarity. These are
historical experimental results, not a current screenshot-only comparison or
an expert ceiling.
[Original experiments](https://arxiv.org/html/2404.07972v2#S4).

## What the 369 original tasks contain

The original collection has eight principal application categories plus
operating-system tasks and cross-application workflows. I recalculated the
following counts from the pinned
[official task list](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/test_all.json);
they sum to 369 and match the paper's Table 10.

| Category | Tasks | Example from the official configurations | What the example tests, in my interpretation |
| --- | ---: | --- | --- |
| Operating system | 24 | Change idle-screen settings | Current environment and system configuration |
| LibreOffice Calc | 47 | Convert accumulated work time and an hourly rate into earnings | Data representation and spreadsheet semantics |
| LibreOffice Impress | 47 | Change slides from landscape to portrait | Document properties and interface operation |
| LibreOffice Writer | 23 | Apply double spacing to the first two paragraphs | Range selection and formatting scope |
| VLC | 17 | Change multiple-instance playback behavior | Application settings and operational knowledge |
| Thunderbird | 15 | Configure a plain-text name and organization signature | Application configuration and text entry |
| Chrome | 46 | Remove Amazon's stored cookies | Browser state and site-data settings |
| VS Code | 23 | Set a user-level wrapping length of 50 | Configuration value and scope |
| GIMP | 26 | Mirror an image horizontally | Image transformation and saved output |
| Cross-application workflows | 101 | Populate reminder recipients from payment records | Selection, state tracking, and information transfer |

The counts are coverage choices. Equal numbers of Calc and Impress tasks do not
imply equal proportions of real user time. Nor does the cross-application
category capture every instance in which an ancillary application appears.

Four tasks show why “computer use” is more than clicking accurately.

### Work time multiplied by an hourly rate

The [earnings task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/libreoffice_calc/357ef137-7eeb-4c80-a3bb-0951f26a8aff.json)
comes from a spreadsheet question. The user has accumulated daily work times
and an hourly rate, but direct multiplication yields the wrong earnings.
The agent must fill the intended result without altering unrelated blank areas.

Spreadsheet time values and numeric hours use different representations. The
operation requires understanding that representation before selecting a formula.
An agent can click the correct cell and still compute a semantically wrong
answer. This task connects application knowledge to an actual saved result.

### A backup that preserves the original content

The [sheet-backup task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/libreoffice_calc/0cecd4f3-74de-457b-ba94-29ad6b5dafb6.json)
asks the agent to rename `Sheet 1` to `LARS Resources`, copy it, place the copy
before another sheet, and apply specified names.

“Make a backup” expands into distinct properties: names, order, duplication,
and preserved cell data. Operations depend on earlier operations, and the agent
must keep track of whether it is editing the source or the copy. The evaluator
must check these properties separately; file existence alone would not suffice.

### Payment records become email recipients

The [tuition-reminder task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/multi_apps/f5c13cdd-205c-4719-a562-348ae5cd1d91.json)
initializes a spreadsheet, a Thunderbird profile, and an email draft. The
requested result is to put the unpaid people's addresses in the recipient field.
The task does not ask the agent to send the email.

Reading its evaluator exposes a useful boundary. The configuration supplies
four selectors for target recipient addresses. The
[checking function](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/desktop_env/evaluators/metrics/general.py)
requires matching elements for each rule; this configuration does not contain a
separate assertion rejecting additional recipients.

Let $G$ be the intended address set and $A$ the actual recipient set. The checks
are closer to testing

$$
G \subseteq A
$$

than to testing $A = G$. Missing addresses can fail while extra addresses need
not fail these assertions. An improved verifier could compare the exact set and
check that the existing email body is preserved, if those are the intended
requirements.

This is a local source inspection of one pinned task, not a benchmark-wide audit
or a demonstrated exploit. It illustrates the difference between user intent
and the properties a verifier actually accepts.

### Select photos and produce an archive

The [photo-selection task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/multi_apps/82e3c869-49f6-4305-a7ce-f3e64a0618e7.json)
asks the agent to identify event photos containing the speaker, copy them to a
specified folder, and create a ZIP archive.

Visual interpretation determines which real files move into the deliverable.
The agent must retain its selection while operating the file system, and the
archive must contain the intended material. A static image question would miss
the execution and packaging errors that this workflow can reveal.

## Why some original tasks are infeasible

The original paper includes 30 tasks labeled infeasible: an application lacks
the requested feature, or the environment lacks the necessary support. An
[official Bluetooth task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/os/b3d4a89c-53f2-4d6b-8b6a-541fb5d205fa.json)
is one example.

These tasks test whether the agent recognizes a limit and stops. In the pinned
[environment implementation](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/desktop_env/desktop_env.py),
an infeasible task receives credit when the final recorded action is `FAIL`.
That signal does not fully evaluate the quality of the explanation or the
completeness of the search for alternatives.

Infeasibility is also version-sensitive. Extensions, alternative methods, or
application changes can invalidate the label. The Verified report acknowledges
such cases. The original count of 30 therefore belongs to the original paper;
it should not be treated as a permanent property of all later releases.

## What Verified repairs

[OSWorld-Verified](https://xlang.ai/blog/osworld-verified) was announced on
July 28, 2025, after the maintainers collected and addressed roughly 300 feedback
items. Those are issue reports, not a count of 300 unusable tasks.

The problems fall into three useful engineering categories:

| Failure source | How it corrupts the measurement | Appropriate repair |
| --- | --- | --- |
| Environment drift | A website changes, blocks automation, or stops serving expected content | Stabilize or replace the dependency and verify initialization |
| Ambiguous or stale task | The instruction and expected outcome diverge | Clarify the requirement or revise the task |
| Evaluator mismatch | Valid formulas or formats fail; incomplete outputs may pass | Expand valid-result acceptance while testing invalid outputs |

The distinction matters. Improving an agent cannot repair a dead link.
Increasing a step budget cannot make an incorrect reference formula correct.
Conversely, accepting more outputs can increase scores without improving the
agent at all. A benchmark repair can change the measurement while leaving the
policy unchanged.

Verified also strengthens parallel evaluation infrastructure and public result
verification. Running tasks across many machines reduces the duration of an
evaluation batch; it does not establish lower latency for a single agent on a
single task.

The broader lesson is that an interactive benchmark's reward signal needs
maintenance. New agents explore paths its authors did not anticipate, and
external systems change. Initial human review cannot guarantee that the same
score has the same meaning indefinitely.

## How 2.0 changes the task-construction problem

The [OSWorld 2.0 paper](https://arxiv.org/html/2606.29537v2) changes the unit of
evaluation to a complete workflow. The project was released on June 26, 2026;
its initial snapshot is `v2026.06.24`, and its first arXiv submission followed on
June 28. Snapshot dates, release dates, and submission dates record different
events.

Version 2.0 contains **108 tasks** across seven professional domains and 21
subcategories. Fewer tasks contain substantially more connected work.
Specialized activities include CAD, video and audio editing, research material
management, and stateful business services. Research and education together
with creative production account for over 40% of the task set. This broadens
coverage while retaining a deliberately curated distribution.

| Dimension | Original OSWorld | OSWorld 2.0 |
| --- | --- | --- |
| Main task count | 369 | 108 |
| Evaluation unit | Often a short operation or workflow | A complete, sustained workflow |
| Workspace | Task files and application state | Coherent profiles, records, messages, and artifacts |
| Applications | General desktop applications | Desktop tools plus professional software and controlled web services |
| Mid-task changes | Not a central design feature | Injected task-relevant messages can update requirements |
| Missing information | Original baselines mainly execute autonomously | Some tasks allow clarification with a bounded simulated user |
| Scoring emphasis | Final functional correctness | Weighted result checkpoints plus strict completion |

The last row is a design emphasis, not a claim that the original environment
could never return fractional rewards. Its paper and code already allow
partial credit. Version 2.0 makes outcome decomposition central to evaluating
long workflows.

### Length must come from dependencies

The authors retain tasks with interdependent work and realistic evidence
distributed through the workspace. A long task can span applications or remain
inside one demanding application. Repeating a simple action or concatenating
unrelated subtasks does not satisfy the same design goal.

About 90% of the retained tasks came from internally trained annotators. They
learned the applications and took responsibility for the task, artifacts, setup,
and evaluator. Practitioner interviews, surveys, and model-generated proposals
also informed collection, but each had limits: cost, missing context, weak
dependencies, or insufficient evaluation detail.

The important sampling boundary remains. These are manually researched and
audited scenarios; “expert-style annotation” does not mean every annotator was
a career expert in the corresponding profession. The collection is not a
production-log sample.
[Task collection](https://arxiv.org/html/2606.29537v2#S2.SS1.SSS1).

### Real applications, controlled service state

Version 2.0 uses 31 self-hosted web services alongside desktop applications.
Email, banking, team chat, and application portals can be reset and inspected.
The agent really operates a computer, but a simulated banking workflow does not
move money in a production bank account.

Artifacts and records form a coherent user state: names, dates, amounts,
identifiers, and previous submissions must agree across sources. Materials are
collected or adapted where possible; synthetic records can be necessary for
privacy, release, or controllability. Authenticity does not imply publishing an
unmodified person's private workspace.

Some tasks inject new messages; some offer a simulated user whose replies are
limited to configured task knowledge. These are two distinct interventions.
A new message changes what the agent must do. A user reply can supply evidence
the agent cannot recover from existing state.
[Environment setup](https://arxiv.org/html/2606.29537v2#S2.SS1.SSS2).

### Read the horizon numbers carefully

The paper reports a median human operation time of about 1.6 hours, compared
with roughly two minutes in the original benchmark. Appendix G.2 explains that
two annotators supplied time ranges, which were converted to midpoints and
combined. These are annotated expected operation times, not direct timing of a
large representative population.

The frequently quoted average of about 318 tool calls corresponds to Opus 4.7
with maximum thinking in the single-action condition; Table 3 reports 318.4.
With batched actions, multiple tool calls can occur in one model turn. **Tool
calls, model turns, mouse clicks, and minimal solution length are different
quantities.** Record the applicable count when comparing efficiency.
[Experimental setup and results](https://arxiv.org/html/2606.29537v2#S3).

## Nine 2.0 tasks that reveal different bottlenecks

The cases below come from the paper's Appendix H and the
[official task showcase](https://osworld-v2.xlang.ai/task-showcase/). The showcase
identifies its task metadata with the June snapshot. Its examples should not be
silently presented as fresh 2.1 runs.

### Task 008: Complete an expense reimbursement

The user attended NeurIPS and gave a talk at Stanford, then asks for registration,
flight, and lodging reimbursement. The agent must consult the instructions,
find supporting receipts and emails, reconcile amounts with bank records,
recover personal information from an earlier report, prepare attachments, and
submit the request.

Each evidence source determines something downstream. The guide determines
categories and required documents. Receipts establish amounts. Bank records
provide reconciliation evidence. The old report supplies unstated employee
details. A new message or clarification can invalidate an earlier interpretation.

![Guidelines, receipts and emails, bank records, and an old report feed the reimbursement form. Updates can require earlier decisions to be checked again before review and submission.](/images/osworld/reimbursement-dependencies.svg)

In a published failure trajectory, the agent spends much of its budget gathering
information and reaches the form near the 500-step limit, without completing
submission. This suggests a specific diagnostic question: does its information
gathering reduce uncertainty about a pending decision, or simply postpone
delivery? The trajectory supports that question; it does not prove that one
particular memory architecture would fix it.

My design inference is that a useful task state should track evidence, unresolved
conflicts, invalidated decisions, and remaining deliverables. A summary saying
“read the receipts” is weaker than a record saying which amount was accepted,
why it was accepted, and which source could still change it.

### Task 035: Review purchase requests as the rules change

The agent reviews requests using a manager's channel messages and direct
messages. Constraints cover budgets, categories, suppliers, dates, and exceptions.
New messages can approve an exception or change a budget or supplier.

The published case shows that noticing an approval does not ensure that every
affected spreadsheet record is updated. This is an invalidation problem: the
agent needs to identify which earlier decisions depended on the superseded rule.
Storing the new message without revisiting dependent decisions is insufficient.

### Task 103: Reconstruct a FreeCAD part

The user supplies an engineering drawing and reference images, requesting a
bracket reconstruction exported as a STEP model. Dimensions and geometric
relationships must survive the conversion from 2D views to a 3D artifact.

This is long-horizon work even if one core application dominates. The published
failure checks the exported file's existence and size without adequately
establishing geometric correctness. Existence is evidence of an export, not
evidence that the hole locations and dimensions satisfy the drawing.

### Task 053: Mask spiders in a video

The user wants black masks over spider monsters in a gameplay video, preserving
duration and minimizing changes elsewhere. The target moves and changes size
across frames.

The case study describes an agent that uses FFmpeg and checks output properties
but still masks the wrong regions at some times or covers too much background.
Tool knowledge and media metadata checks leave the task's temporal and spatial
requirements unresolved. Verification needs evidence at the granularity of the
requested edit.

### Task 052: Select a hotel room behind a moving popup

The agent must select the Deluxe Suite while leaving personal-information entry
to the user. A moving promotional popup interferes with interaction.

The screenshot captures a close button at one location; the eventual click may
arrive after it moves. The task stresses observation-to-action latency as well
as spatial localization. This is a different problem from purchase-order
updates: the hotel case changes interface geometry, whereas the purchase case
changes the semantic basis of decisions.

A deliberately moving popup can expose an architectural weakness. Its presence
does not establish how frequent that weakness is in everyday user work.

### Task 024: Ask for missing evidence before submitting

The user requests a DS-2019 application and asks the agent to review documents
before submission. The provided deposit certificate shows USD 12,000, while the
task's requirements call for USD 18,000. The agent should detect the mismatch and
ask the simulated user for adequate evidence.

This evaluates whether the agent can recognize when its current evidence is
insufficient. Asking a specific question about the shortfall is more useful
than a generic request for more documents. Receiving a new file does not end
the verification obligation; the replacement must resolve the discrepancy.
These amounts describe the benchmark scenario, not a general statement about
real visa requirements.

The paper also documents a trajectory that handles this challenge but loses
official score because of evaluator canonicalization. That example matters:
task score and successful handling of a particular phenomenon can disagree.
The authors' exposure analysis helps separate those claims.
[Appendices G.3 and H.2.3](https://arxiv.org/html/2606.29537v2#A7.SS3).

### Task 004: Extend slides while preserving earlier work

The agent must format newly added Meta Chain-of-Thought slides consistently
with an existing deck: fonts, footers, and alignment. Earlier slides must remain
unchanged. The public scoring description includes preservation as a gate.

The existing artifact supplies the specification. A good edit must infer its
style and restrict the modification scope. A verifier that rewards attractive
new slides while ignoring damage to old ones misses an explicit requirement.

### Task 055: Learn an editing procedure from a reference video

The agent uses Shotcut to recreate a reference edit from source clips, including
transitions, split screens, and animated captions. Tutorials and the reference
provide the procedure.

Extracted frames can show what appears at a moment while losing transition
duration, movement, or edit timing. The case stresses learning a process from
temporal evidence. A set of recognizable visual elements is not equivalent to
the intended sequence.

### Task 098: Map source documents into a guided form

The agent receives a photo, passport, DS-2019, basic information, and a guide for
completing a DS-160 form. It must align guide instructions with source evidence
and the correct conditional fields, adjusting document files when needed.

The sources and destination have different structures. The agent must maintain
a mapping between them rather than copy nearby text. The same structural
problem appears in insurance, reimbursement, and enterprise application forms.

Together, these cases motivate different interventions: better geometry checks,
temporal perception, dependency invalidation, evidence-aware questioning,
artifact-preservation checks, and delivery planning. One generic “reason more”
instruction does not specify which mechanism should improve.

## Challenge tags improve diagnosis when exposure is recorded

The 2.0 authors annotate ten challenge phenomena. The tags overlap: a task can
involve several, so their counts must not be summed into a task total.
[Table 2](https://arxiv.org/html/2606.29537v2#S2.SS2.SSS2).

| Phenomenon | Tasks | What the tagged difficulty concerns |
| --- | ---: | --- |
| Cross-source reasoning | 46 | Reconcile evidence across files, services, and records |
| Visual-spatial precision | 45 | Geometry, position, layout, or fine visual detail |
| Implicit-state inference | 43 | Recover relevant state omitted from the instruction |
| Multi-item state tracking | 43 | Maintain decisions over multiple records or objects |
| Conflict disambiguation | 39 | Resolve conflicting, stale, or distracting information |
| Multimodal editing | 30 | Modify image, video, audio, or 3D artifacts |
| Tutorial following | 22 | Extract procedures from guides or reference work |
| Dynamic environment | 10 | Respond to messages that change requirements |
| Streaming interaction | 6 | Act while the interface continues changing |
| Proactive interaction | 6 | Ask when evidence or constraints remain insufficient |

An application label says where a failure occurred. A phenomenon label helps
describe what the agent needed to do. But a low score on a tagged task does not
prove failure on that phenomenon. An agent can fail earlier and never reach it.

The paper distinguishes **handled**, **blocked**, and **untested** exposure.
For analysis I would report both the fraction of relevant episodes reaching a
challenge and the handled fraction among reached episodes. The latter is
conditional on surviving earlier parts of the workflow. Reporting it alone can
favor an agent that reaches only easier cases.

Even that breakdown is diagnostic rather than causal: holding other conditions
fixed and intervening on the relevant mechanism is still necessary to establish
why performance changes.

## How partial score differs from complete delivery

Version 2.0 has an average of 27.25 scoring checkpoints per task. These check
the final environment state and artifacts, permitting alternative valid paths
rather than requiring one checkpoint order. Functional checks are preferred;
bounded model judgments handle some open-ended text and visual requirements.

The paper reports model judgments contributing 11.53% of total score, with no
task relying on them for more than 50%. Those percentages concern score weight,
not the share of tasks or execution steps judged by a model.
[Evaluation protocol](https://arxiv.org/html/2606.29537v2#S2.SS1.SSS3).

### Preserve the task-level aggregation

The public
[result-output contract](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/docs/EVALUATE_RESULT_JSON.md)
describes criterion scores and weights, with normalized weights summing to one.
For a task $i$, the additive form is

$$
\begin{gathered}
r_i = \sum_{j=1}^{K_i} w_{ij}c_{ij},\\
\sum_j w_{ij}=1,\quad c_{ij}\in[0,1].
\end{gathered}
$$

This describes the public decomposition contract; it is not a claim that every
gated task lacks gates or other task-specific logic. Inspect the task's actual
evaluator to understand its aggregation.

For a single run on each of $N$ tasks, average partial score and strict
completion are different quantities:

$$
\bar r = \frac{1}{N}\sum_i r_i,
\qquad
C = \frac{1}{N}\sum_i \mathbf{1}[r_i=1].
$$

The paper defines binary completion at a partial score of 1.00. Pooling every
checkpoint across tasks instead would implicitly favor tasks with more
checkpoints; it is not the same task-weighted statistic.

Here is a **constructed teaching example**, not the official Task 008 rubric:

| Reimbursement criterion | Weight | Satisfied? |
| --- | ---: | --- |
| Amounts and categories correct | 0.30 | Yes |
| Identity fields correct | 0.20 | Yes |
| Required attachments present | 0.30 | Yes |
| Final submission completed | 0.20 | No |

The partial score is 0.80 and strict completion is zero. If preservation is a
gate, as in the slide example, violating it may instead prevent otherwise
correct edits from earning their expected credit. That dependency has to be
read from the scoring design.

### The headline numbers belong to one experimental contract

The 2.0 paper's v2 reports Opus 4.8 with maximum thinking and batched tool calls
at **20.6% strict completion and 54.8% mean partial score**. The setup uses
screenshot observations, a 500-step budget, and the **`v2026.06.24` release**.
Claude uses its native computer-use tool; other evaluated families use
PyAutoGUI actions. The paper specifies additional generation, pause, and
infrastructure conditions.
[Section 3.1 and Table 3](https://arxiv.org/html/2606.29537v2#S3.SS1).

The 54.8% is an average normalized task score. It does not mean that 54.8% of
the tasks were completed, or that each task was halfway done. The two aggregate
metrics also do not reveal the distribution of scores among incomplete tasks.

These results expose a delivery gap in that setup. They are not October's live
ranking, a new 2.1 measurement, or a controlled comparison with the 2024
12.24% baseline. Task distributions, models, budgets, and observation conditions
all differ.

## A more detailed outcome score is still not action credit

Suppose the final score is 0.7. The checkpoint breakdown tells us which result
properties hold. It may not tell us whether the decision at step 50 caused the
missed submission at step 300, whether a later recovery saved an earlier error,
or which action deserves a policy update.

This separates three questions:

| Signal | What it can reveal | What remains unresolved |
| --- | --- | --- |
| Final completion | Whether all scored requirements hold | Which decisions caused the outcome |
| Final checkpoint decomposition | Which result properties hold | When they became true and which actions established them |
| Process observations or counterfactual replay | Where progress or failure appears; sometimes the value of a local alternative | Confounding, coverage, and validity of the alternative continuation |

For an episodic policy-gradient update, a terminal reward can enter an estimator
of the form

$$
\widehat g = \sum_t \nabla_\theta\log\pi_\theta(a_t\mid h_t)
\bigl(r_T-b(h_t)\bigr),
$$

where $h_t$ is the observed history and $b$ is an action-independent baseline.
The equation is an explanatory RL model, not an OSWorld training algorithm.
The terminal outcome enters every sampled decision's return. Having more final
checkpoint values does not automatically supply causal, action-specific labels.

Intermediate evaluation requires additional engineering. In the original
[`evaluate()` implementation](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/desktop_env/desktop_env.py),
`postconfig` executes before scoring and can perform setup-style operations.
Do not assume evaluation is a side-effect-free sensor that can be called after
every action. A training system should evaluate cloned states, verify
idempotence and side effects, and keep privileged answers out of agent
observations.

Nor is a change in checkpoint score necessarily monotonic progress. A field can
be filled correctly and later overwritten; a temporary state can satisfy a
check while the final deliverable remains wrong. If using score changes for
reward shaping, distinguish an engineered training objective from the
benchmark's terminal metric and test the resulting policy against the latter.

## Verification needs both acceptance and rejection tests

The 2.0 construction pipeline includes scoring tests, independent human
completion and cross-review, and multiple agent rollouts. Disagreement prompts
revision; persistent ambiguity or infeasibility can lead to task removal.
[Quality assurance](https://arxiv.org/html/2606.29537v2#S2.SS1.SSS4).

This addresses two opposing verifier failures:

1. **False acceptance:** an incomplete or invalid result earns credit.
2. **False rejection:** a valid alternative loses credit because the checker is
   too rigid or the judge disagrees with acceptable evidence.

The recipient-set example suggests an invalid-output test: include every target
address plus an unwanted one. Spreadsheet tasks suggest valid-output tests:
use a different but equivalent formula or harmless formatting variation.
Testing only a reference solution establishes too little.

Completion and safe execution also need separate evidence. The paper includes
diagnostic audits of side effects and forbidden shortcuts, including credential
exposure during a GitLab workflow. A successful output cannot establish that
the agent avoided unwanted changes throughout execution.
[Safety analysis](https://arxiv.org/html/2606.29537v2#S4.SS2).

The information boundary matters here. Evaluators need access to internal
state. Evaluated agents should receive only the channels authorized by the
protocol. Reading evaluator files, reference artifacts, or service databases
can turn a UI task into a different problem. A result therefore needs an
explicit tool-permission record as well as a completion score.

## Where 2.1, Human, and Pro fit

The projects share a task lineage but do not form one sequence of interchangeable
leaderboards.

![The main line runs from original OSWorld to Verified, 2.0, and the 2.1 repair release. Human studies efficiency and Pro studies process diagnosis as separate research branches.](/images/osworld/version-map.svg)

### 2.1 pins a coordinated release

The official repository lists **September 16, 2026** as the 2.1 release date and
recommends it for current runs. The manifest still lists 108 task files.
[Repository release history](https://github.com/xlang-ai/OSWorld-V2),
[2.1 manifest](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/benchmark_releases/osworld-v2.1.json).

The important change is a coordinated version contract:

| Component | Why it must match the release |
| --- | --- |
| Environment and runner code | Implements the observation, action, and evaluation protocol |
| Task implementations | Defines setup and checks for the intended tasks |
| Complete assets | Supplies the exact input and reference files |
| Website source and deployment | Supplies the required service behavior and initial records |
| VM image and runtime | Supplies application versions and machine behavior |
| Task hashes and run provenance | Makes the selected inputs and execution conditions inspectable |

The 2.1 task implementations and complete assets are distributed through gated
Hugging Face datasets. The public runtime-asset dataset contains only files
needed for anonymous browser-facing access and is not the complete snapshot.
The June websites are no longer hosted at the old endpoint; reproducing that
release requires self-hosting its matching website source. A floating `main`
checkout plus unrelated assets is not a faithful replay.
[Release management](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/benchmark_releases/README.md).

Read the manifest's validation boundaries as carefully as its component pins.
It explicitly does not claim a full 108-task agent evaluation. It records that
Docker image bytes and runtime digests were verified without a successful Docker
VM boot, and that unmodified Task029 retained an observed setup timeout.
A release tag or a passing smoke test cannot establish that every task works.

### Human adds an efficiency reference

[OSWorld-Human](https://arxiv.org/abs/2506.16042), first submitted in June 2025,
adds human reference trajectories and examines extra agent steps and latency
from planning and reflection. It asks how efficiently an agent completes work,
not just whether it eventually succeeds.

Action grouping affects that comparison. A batched action and an atomic mouse
operation cannot be counted as interchangeable units. Reference trajectories
also contain solutions to evaluation tasks; the
[authors' repository](https://github.com/WukLab/osworld-human) warns against
using them as training data for the same evaluation.

### Pro adds process diagnosis

[OSWorld-Pro](https://arxiv.org/abs/2609.24890), submitted on September 21, 2026,
describes over 300 tasks, over 2,800 subgoals, and more than 67,000 human
annotations. Human-aligned model judges evaluate progress through dependent
subgoals, helping distinguish input errors, click errors, and actions irrelevant
to the current subgoal.

This is a research extension of process evaluation, not a replacement name for
2.1. Subgoal fulfillment also does not, by itself, solve the causal credit
assignment problem described above.

## What I would test for a long-horizon agent

The cases suggest mechanisms that can be implemented and compared. They do not
prove that a proposed memory, verification, or RL method will improve the
benchmark. The following is an **unrun experimental design**.

### Begin with a precise evaluation contract

Treat a result as depending on

$$
\operatorname{Eval}(\mathcal D_v, E_v, \pi_{\theta,H}, O, A, B, V_v),
$$

where $\mathcal D_v$ is the task set, $E_v$ the released environment,
$\pi_{\theta,H}$ the model inside agent architecture $H$, $O$ and $A$ the
observation and action protocols, $B$ the budget, and $V_v$ the verifier.
This is a reporting convention, not a new capability metric.

Record those inputs before attributing a gain to the model. A stronger model,
a longer budget, direct database access, improved initialization, and a corrected
verifier can all change the score through different mechanisms.

### Separate state maintenance from verification

I would first run a factorial comparison on the same held-out tasks:

| Condition | Task-state maintenance | Result verification |
| --- | --- | --- |
| Baseline | Existing context strategy | Existing checking behavior |
| Explicit state | Evidence and constraint records with invalidation | Baseline verification |
| Verification loop | Baseline context strategy | Artifact-specific checks and repair |
| Combined | Explicit state | Verification loop |

The state record should identify each fact's source, decision, dependencies,
and validity. A new budget message should mark affected purchase decisions for
rechecking. Verification should inspect the requested property: CAD geometry,
recipient sets, protected slides, or actual submission state. The agent's checks
must use permitted evidence rather than the privileged benchmark rubric.

Hold the model, release, permissions, and total interaction budget fixed.
Count verification actions inside that budget. Use matched tasks, predefined
repeats, and task-level confidence intervals; pair environment seeds or initial
states where the setup permits. Report strict completion, mean partial score,
tool calls, model turns, tokens, and wall-clock time. A second comparison at a
fixed time budget can reveal latency tradeoffs hidden by a turn limit.

Mechanism-specific observations should accompany the scores. Did new messages
trigger rechecking of every affected row? Did the reimbursement agent reach
delivery earlier? Did geometry checking detect a real error, and did the repair
introduce a different one? These distinguish a useful intervention from a
larger amount of activity.

### Measure recovery as well as local correctness

A small calculation explains why good local performance can coexist with poor
delivery. Suppose a teaching workflow has $H$ required stages, each independently
correct with probability $p$, with no recovery. Then

$$
P(\text{complete})=p^H.
$$

At $p=0.98$, complete success is about 66.8% over 20 stages and 13.3% over 100.
These are calculated values from an independence model, **not an OSWorld fit**.
Real errors can be correlated, and agents can recover. The calculation explains
why an average local-success rate does not determine delivery reliability.

Recovery changes the structure. If a detected mistake is repaired before it
propagates, one bad action need not ruin the task. A useful comparison therefore
records missed errors, detected errors, successful repairs, and new errors
introduced by repair. “More reasoning” does not tell us which of these changed.

### Separate training improvements from replaying the test

Do not train on the same task solutions and report the resulting test gain as
generalization. Even changing names and amounts may leave the workflow template
and solution strategy intact. Depending on the research question, useful splits
can hold out workflow families, input-artifact sources, initialization states,
or interface arrangements.

Evaluate learning separately from runtime assistance. An external memory or
verifier can improve the deployed agent without changing the model parameters;
an RL update can change the policy without demonstrating a benefit from that
runtime mechanism. Both can be useful, but they answer different questions.

The original suite and shorter independent workflows can support development,
with 2.0 reserved for a defined held-out evaluation. Many specialized 2.0 tools
appear in only one to three tasks, so successful examples are weak evidence of
mastering an entire profession. Document the split and exposure rather than
relying on a single average.

## What this benchmark teaches us about computer-use progress

The strongest thread through OSWorld's history is the work required to make
completion observable. First the authors built executable computer tasks.
Then they repaired mismatches between intent, environment, and checks. Next
they expanded the unit of work until maintaining evidence and constraints became
a central difficulty. Versioned releases make those measurements reproducible.

For an agent builder, the next useful question is concrete: **after many files,
windows, and updates, does the agent still know what the evidence supports,
which decisions need revision, and what remains to be delivered?**

OSWorld offers tasks on which to test that question. Answering it well requires
preserving the experiment's conditions, inspecting its verifier, and measuring
complete work alongside partial progress and execution cost.

## Primary sources and task implementations

The original task and evaluator links below use a pinned repository commit.
Paper links specify the versions read. The 2.1 documentation uses its release
tag; the case studies and reported 2.0 results belong to the June release.

1. [OSWorld original paper, v2, May 30, 2024](https://arxiv.org/html/2404.07972v2).
2. [OSWorld repository and release history](https://github.com/xlang-ai/OSWorld).
3. [Original task list](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/test_all.json).
4. [Sheet rename and backup task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/libreoffice_calc/0cecd4f3-74de-457b-ba94-29ad6b5dafb6.json).
5. [Payment records to email recipients](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/multi_apps/f5c13cdd-205c-4719-a562-348ae5cd1d91.json).
6. [Work time and hourly earnings](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/libreoffice_calc/357ef137-7eeb-4c80-a3bb-0951f26a8aff.json).
7. [Photo selection and archive task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/multi_apps/82e3c869-49f6-4305-a7ce-f3e64a0618e7.json).
8. [Writer paragraph-spacing task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/libreoffice_writer/0810415c-bde4-4443-9047-d5f70165a697.json).
9. [Impress slide-orientation task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/libreoffice_impress/ce88f674-ab7a-43da-9201-468d38539e4a.json).
10. [VLC multiple-instance task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/vlc/f3977615-2b45-4ac5-8bba-80c17dbe2a37.json).
11. [Thunderbird signature task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/thunderbird/3f28fe4f-5d9d-4994-a456-efd78cfae1a3.json).
12. [Chrome cookie-removal task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/chrome/7b6c7e24-c58a-49fc-a5bb-d57b80e5b4c3.json).
13. [Bluetooth infeasibility task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/os/b3d4a89c-53f2-4d6b-8b6a-541fb5d205fa.json).
14. [Accessibility-tree checking implementation](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/desktop_env/evaluators/metrics/general.py).
15. [Environment and evaluation implementation](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/desktop_env/desktop_env.py).
16. [OSWorld-Verified announcement](https://xlang.ai/blog/osworld-verified).
17. [OSWorld 2.0 paper, v2, July 13, 2026](https://arxiv.org/html/2606.29537v2).
18. [OSWorld-V2 repository and release history](https://github.com/xlang-ai/OSWorld-V2).
19. [OSWorld 2.0 submission record](https://arxiv.org/abs/2606.29537).
20. [Criterion scores and weights output contract](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/docs/EVALUATE_RESULT_JSON.md).
21. [OSWorld 2.0 task showcase](https://osworld-v2.xlang.ai/task-showcase/) and [project case studies](https://osworld-v2.xlang.ai/).
22. [2.1 component manifest and verification limitations](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/benchmark_releases/osworld-v2.1.json).
23. [Release-management contract](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/benchmark_releases/README.md).
24. [OSWorld-Human paper](https://arxiv.org/abs/2506.16042).
25. [OSWorld-Human repository](https://github.com/WukLab/osworld-human) and [authors' efficiency discussion](https://mlsys.wuklab.io/posts/oshuman/).
26. [OSWorld-Pro paper](https://arxiv.org/abs/2609.24890).
27. [VS Code wrapping-length task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/vs_code/276cc624-87ea-4f08-ab93-f770e3790175.json).
28. [GIMP horizontal-mirror task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/gimp/72f83cdc-bf76-4531-9a1b-eb893a13f8aa.json).
29. [Idle-screen dimming task](https://github.com/xlang-ai/OSWorld/blob/b138d348256078fa634fc3b73567a7337c793e6b/evaluation_examples/examples/os/bedcedc4-4d72-425e-ad62-21960b11fe0d.json).

Related implementation notes on this blog:
[a reproducible agent RL stack](/posts/2026-08-14-agent-rl-reproducible-closed-loop/),
[trajectory-based evaluation difficulty](/posts/2026-05-28-agent-eval-difficulty-trajectory-constraints/),
and [learning from failed agent rollouts](/posts/2026-09-20-negative-signals-agent-rft/).
