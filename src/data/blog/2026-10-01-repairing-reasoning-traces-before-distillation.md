---
author: Jing Lu
pubDatetime: 2026-10-01T16:00:00-07:00
title: "How Should We Repair Reasoning Traces Before Distillation?"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Post Training
  - Reinforcement Learning
  - Evaluation
  - ML Engineering
description: "A map of 26 research papers on reasoning distillation, trajectory correction, and student compatibility: where each method acts, what it costs to implement, and an experiment to test how much of an RL teacher trace to rewrite."
---

Suppose a large model and a small model have been trained on the same task
data. The large model has also learned a stronger policy through reinforcement
learning. We collect its reasoning trajectories and use supervised fine-tuning
to distill them into the small model.

Some trajectories contain behavior we would rather not teach: an incorrect
intermediate calculation, a repeated check, a premature conclusion, or a tool
call that violates a task constraint. Asking a stronger teacher to fix those
parts seems straightforward. Yet the corrected data can produce a worse
student.

One explanation is that the inserted correction sounds different from the
surrounding trace. Another is that it changes an intermediate result while
leaving an incompatible continuation intact. A third is that the entire
teacher solution is too far from the student's usable reasoning patterns.
These explanations imply different interventions. Calling all of them a
“style mismatch” makes the research question harder to answer.

My starting conclusion is that **the literature does not establish one
universally best editing rule**. It gives us several mechanisms, useful
baselines, and increasingly direct evidence that better-looking reasoning is
not always better supervision. The practical question is how to locate our
problem within that larger picture.

This post maps 26 relevant papers available by **October 1, 2026**, compares
their implementation requirements, and proposes a controlled study of repair
scope. It is a literature synthesis and research design. **No new model
training results are reported here**, and several of the recent methods are
preprints. The difficulty ratings are my engineering estimates relative to an
existing offline SFT pipeline, not measured runtimes or a ranking of scientific
merit.

## Table of contents

## First separate the decisions in the pipeline

A teacher trained with RL does not make the student's training RL. If the
student learns by predicting fixed teacher tokens, its update is still SFT.
Likewise, collecting student rollouts does not by itself specify whether the
update uses cross-entropy, distribution matching, preference optimization,
or task rewards.

The complete pipeline has more choices than “SFT versus RL” suggests:

```text
0. Train or choose a teacher
          ↓
1. Collect teacher, student, or mixed trajectories
          ↓
2. Verify outcomes and locate problematic behavior
          ↓
3. Keep, edit, regenerate, or discard a target
          ↓
4. Decide which tokens receive supervision
          ↓
5. Update the student: SFT, KL, preferences, or RL
          ↓
6. Evaluate independent student behavior
          └── optionally refresh the data at step 1
```

Four choices help place almost any method in this map: **whose prefixes it
uses**, **what signal the teacher supplies**, **how long a target it constructs**,
and **whether the data refreshes during training**. A prefix is everything the
model has seen or generated before its next decision. In a tool-using agent,
that includes actual environment observations.

The following families overlap. An expert takeover can use both selective loss
masking and SFT. A trajectory rewrite can be followed by KL distillation.
These are components that can be combined, not mutually exclusive schools.

## The research landscape, by intervention point

The “effect” column describes the mechanism and evidence in the cited work.
Results come from different models, tasks, budgets, and metrics; they do not
form a common leaderboard.

| Research family                            | Where it acts                         | Representative work                                                                                                                                                                                                                                                       | What it changes or shows                                                                                                      | Implementation difficulty                                                  | Main limitation                                                              |
| ------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Train a teacher to teach                   | 0: teacher objective                  | [RLT](https://arxiv.org/abs/2506.08388)                                                                                                                                                                                                                                   | Rewards explanations using student feedback; reports effective downstream distillation from a relatively small teaching model | High: teacher RL and student scoring                                       | Teaching rewards remain proxies for eventual learning                        |
| Select acceptable demonstrations           | 1–2: generation and verification      | [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [STaR](https://arxiv.org/abs/2203.14465), [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)                                                                                                                  | Uses successful samples, iterative rationalization, or process-based ranking to improve the supervision pool                  | Low–medium with an existing verifier; high to train a new process verifier | Filtering can remove the hardest problems and does not choose repair scope   |
| Change the representation                  | 3: wording, structure, length         | [Quality-Utility Paradox](https://arxiv.org/abs/2606.16152), [Compress-Distill](https://arxiv.org/abs/2606.05988), [PART](https://arxiv.org/abs/2510.11545)                                                                                                               | Shows that quality, length, information content, and downstream learning can move differently                                 | Low–medium to implement; difficult to isolate causes                       | Wording changes often also change reasoning structure or information density |
| Construct coherent corrections             | 3: step or trajectory targets         | [TRD](https://arxiv.org/abs/2606.08432), [SOPD](https://arxiv.org/abs/2608.16333), [SuperCorrect](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0967d7c8b171dd81b77c43067c02bebf-Abstract-Conference.html)                                                      | Supplies a repair path or structured reasoning target beyond a single next-token correction                                   | Medium–high: target generation, validation, sometimes logits or DPO        | A valid target can still be hard for the student to learn                    |
| Cover states the student visits            | 1 and 6: rollout source and refresh   | [GKD](https://arxiv.org/abs/2306.13649), [MiniLLM](https://arxiv.org/abs/2306.08543), [OEC](https://arxiv.org/abs/2512.14895), [Motab](https://arxiv.org/abs/2605.19433), [Revisiting DAgger](https://arxiv.org/abs/2605.12913), [RISE](https://arxiv.org/abs/2407.18219) | Moves supervision toward student-generated prefixes, interventions, or repeated attempts                                      | Medium–high; interactive environments add substantial work                 | More relevant states do not guarantee coherent or learnable targets          |
| Change the distribution-matching signal    | 5: weights and soft targets           | [DisCorD](https://arxiv.org/abs/2605.14071), [TAID](https://arxiv.org/abs/2501.16937), [TIDE](https://arxiv.org/abs/2608.09836), [TOPD](https://arxiv.org/abs/2606.00305)                                                                                                 | Reweights offline tokens, interpolates distributions, or adds guidance about future continuations                             | Medium–high: probability access, custom losses, often online rollouts      | Weighting alone cannot make a contradictory trace valid                      |
| Choose which tokens to learn               | 4: supervision mask                   | [STeP](https://arxiv.org/abs/2505.20023), [OEC](https://arxiv.org/abs/2512.14895)                                                                                                                                                                                         | Retains useful context while supervising selected repair or expert tokens                                                     | Medium: reliable span labels and loss masks                                | Learning to recover after an error differs from preventing it                |
| Learn a preference between decisions       | 3 and 5: paired targets and objective | [Step-DPO](https://arxiv.org/abs/2406.18629), [SuperCorrect](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0967d7c8b171dd81b77c43067c02bebf-Abstract-Conference.html)                                                                                           | Trains the model to prefer a correct step or correction over an alternative                                                   | Medium–high: valid pairs, reference model, preference training             | Bad pairs or uncontrolled length differences can mislead training            |
| Optimize outcomes under the student policy | 5–6: rewards and interaction          | [SCoRe](https://arxiv.org/abs/2409.12917), [LUFFY](https://arxiv.org/abs/2504.14945), [Student-Centered Distillation](https://arxiv.org/abs/2509.14257)                                                                                                                   | Uses multi-turn RL, mixed teacher/student guidance, or SFT followed by local RL                                               | High: rollout infrastructure, rewards, policy optimization                 | Larger systems obscure whether editing itself caused the gain                |

A useful boundary case is
[Co-Evolving Harnesses and Models](https://arxiv.org/abs/2609.09134). In its
enterprise-agent settings, imitating full expert trajectories can disrupt the
student's adaptation to its execution harness, while corrections of specific
failure turns help. This is evidence about model–harness compatibility in that
setting; it is not a direct experiment on editing a large model's mathematical
reasoning trace.

The table also exposes a common comparison mistake. R1-style rejection
sampling, a process reward model, and student-policy RL solve different
subproblems. A gain in selecting the best of many answers is not a gain in a
distilled student's single-attempt accuracy. A full SFT-plus-RL system beating
a baseline does not establish that its correction text was the decisive
ingredient.

## What the closest papers actually tell us

For the specific problem of modifying a stronger RL model's traces, I would
read the representation and correction papers first, then the papers about
student state coverage and teaching objectives.

### Higher quality does not settle learning utility

The [Quality-Utility Paradox](https://arxiv.org/abs/2606.16152) is a close
neighbor. It studies stronger-model refinement of student-native mathematical
traces and reports that higher-scoring refined data can train worse students.
Its style-aligned refinement preserves more of the student's original
reasoning form and recovers utility. That motivates a compatibility hypothesis,
but its starting distribution differs from our fixed large-teacher traces.
Preserving the original teacher's style and adapting a trace to the student's
style are two separate interventions.

[Compress-Distill](https://arxiv.org/abs/2606.05988) illustrates another tradeoff:
its reported training speedups of 2.0–7.6 times come with lower accuracy than
raw traces across its tested student-size and teacher combinations. Compression
may be useful under a compute constraint without being the best recipe for
unconstrained accuracy.

[PART](https://arxiv.org/abs/2510.11545) deliberately reformulates reasoning to
impede distillation while preserving information. It is diagnostic evidence,
not a recipe I would adopt to improve a student. Together these papers make
“the explanation still contains the answer” an inadequate acceptance test.

### A repair needs a coherent future

[Trajectory-Refined Distillation](https://arxiv.org/abs/2606.08432) revises
student rollouts before applying distillation. Its original recipe includes
teacher distribution matching on the refined paths. Copying its editing idea
and training with hard-label cross-entropy would be a useful **text-only
variant**, not a reproduction of the full method. Also, a teacher-transformed
student sample is no longer literally a sample from the unmodified student
policy.

[SOPD](https://arxiv.org/abs/2608.16333) provides teacher-generated step targets
conditioned on prefixes from a full student rollout. These targets are
independent alternatives at their respective prefixes. They should not be
concatenated into a fictional single trajectory. Its efficient implementation
uses attention isolation so each target sees the appropriate prefix and its
own generated tokens. The authors report a 13.4 percentage-point improvement
over vanilla OPD on ALFWorld; that result belongs to their experimental setup.

For agents, [OEC](https://arxiv.org/abs/2512.14895) instead lets a student act
before switching to an expert at a sampled turn, validates the resulting
execution, and supervises the expert portion. This is a concrete precedent
for suffix takeover. A changed tool action must be executed to obtain new
observations; textually editing an action while retaining the old observations
does not produce a valid demonstration.

### State coverage and teaching quality are different controls

[GKD](https://arxiv.org/abs/2306.13649) and
[MiniLLM](https://arxiv.org/abs/2306.08543) place distributional supervision on
student-generated trajectories; MiniLLM emphasizes reverse-KL optimization.
[Motab](https://arxiv.org/abs/2605.19433) studies monitoring, backtracking, and
teacher intervention when student reasoning strays.
[Revisiting DAgger](https://arxiv.org/abs/2605.12913) explores iterative mixed
teacher/student interaction for LLM agents. Their shared relevance is the
choice of training contexts. They do not imply that arbitrary edits of those
contexts will be useful.

[RLT](https://arxiv.org/abs/2506.08388) changes the source of supervision more
fundamentally. It trains a teacher, given a problem and solution, to write
explanations rewarded through student feedback. That already occupies part of
the “optimize the teacher for the learner” research space. Its reward evaluates
the student's response to an explanation; it does not retrain a student on
every candidate explanation to measure its eventual generalization benefit.
An editor trained with student feedback would need to explain how it differs
from this existing teaching objective.

### Recovery requires the right training context

[STeP](https://arxiv.org/abs/2505.20023) uses reflected trajectories and partial
masking to avoid treating selected erroneous or suboptimal behavior as positive
labels. [RISE](https://arxiv.org/abs/2407.18219) studies iterative training for
multi-turn improvement. Google's
[SCoRe](https://arxiv.org/abs/2409.12917) identifies limitations in offline
self-correction training and uses a two-stage RL approach for multi-turn
correction. Those results motivate measuring recovery under the student's own
errors, rather than assuming that imitation of polished demonstrations is
sufficient.

There is also an agent method called SCoRe in
[Student-Centered Distillation](https://arxiv.org/abs/2509.14257), previously
titled _From Correction to Mastery_. It combines correction-oriented SFT with
short-horizon RL from valid prefixes. It is a different paper from Google's
2024 SCoRe. Its combined results should not be attributed to correction SFT
alone.

## What would each option require in practice?

This table assumes a working offline SFT trainer. “Text-only” means no access
to the teacher's token probabilities is required; it does not mean that the
method needs no verifier, student scoring, or environment. Costs include data
construction, not just the optimizer step.

| Option                              | Teacher access                                                             | New sampling or execution                          | Changes beyond ordinary SFT                                                  | Relative effort                        |
| ----------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------- |
| Filter and resample teacher traces  | Text-only                                                                  | More teacher candidates                            | Verifier, provenance, coverage accounting                                    | Low                                    |
| Local patch or suffix rewrite       | Text-only                                                                  | Teacher edits; replay changed agent actions        | Span tracking and independent validation                                     | Low–medium for text; higher for agents |
| Preserve style while repairing      | Text-only                                                                  | Paired edit candidates                             | Constraints on structure, length, and edit scope                             | Medium; causal attribution is harder   |
| STeP-style partial masking          | Text-only                                                                  | Correction or reflection data                      | Label masks at known spans                                                   | Medium                                 |
| OEC-style expert takeover           | Text-only                                                                  | Student/expert execution and verification          | Model switching, teacher-only labels, data refresh                           | Medium–high                            |
| SOPD-style independent step targets | Text-only                                                                  | Student rollouts plus many teacher steps           | Separate prefix/target examples; specialized attention for efficient packing | Medium–high                            |
| DisCorD                             | Teacher token log-probabilities                                            | Fixed offline traces; teacher scores can be cached | Bounded token weighting                                                      | Medium                                 |
| TAID                                | Teacher and student logits with aligned token support                      | Depends on the data pipeline                       | Interpolated targets and a training schedule                                 | Medium–high                            |
| GKD / MiniLLM / TIDE                | Teacher probabilities or distributions as required by the objective        | Student rollouts during training                   | Rollout/trainer coordination and distributional objectives                   | High without an existing OPD stack     |
| TRD, full recipe                    | Teacher text and distributions                                             | Student sampling plus trajectory refinement        | Refinement loop and KL distillation                                          | High                                   |
| TOPD                                | Distributional feedback plus short continuations                           | Additional teacher/student lookahead               | Future-window comparison and aligned guidance                                | High                                   |
| Step-DPO / SuperCorrect             | Text-only for target construction                                          | Verified preferred/rejected examples               | Preference training and reference-model handling                             | Medium–high                            |
| RLT / SCoRe / LUFFY                 | Depends on the method; rewards alone are not enough to implement all three | Teacher or student RL rollouts                     | Policy optimization, reward design, evaluation                               | High                                   |

[DisCorD](https://arxiv.org/abs/2605.14071) is particularly relevant when the
offline dataset is fixed: it uses a bounded transformation of student/teacher
token-probability ratios to change training weights. This does not generate
missing student states or establish an unbiased reconstruction of on-policy
training. [TAID](https://arxiv.org/abs/2501.16937) instead interpolates student
and teacher logits over training.
[TIDE](https://arxiv.org/abs/2608.09836) changes the learning signal under strong
distribution mismatch, while [TOPD](https://arxiv.org/abs/2606.00305) uses short
future continuations to guide alignment. These are alternatives or complements
to changing the trace text.

For a text-only teacher API, I would start with verified resampling, suffix
repair, masking, and independent step targets. For an existing on-policy
distillation platform, distributional baselines become more affordable. The
same method can be a small extension in one lab and a major infrastructure
project in another.

## Why a small edit can cause a large training problem

Consider a deliberately simple, hand-written trace:

```text
We need 12 × 15.
12 × 10 = 120, and 12 × 5 = 50.
Adding the two terms gives 170.
Therefore the answer is 170.
```

Changing `50` to `60` repairs one calculation. It leaves the next two lines
wrong. Replacing only that line with a formal algebraic derivation might also
introduce a different explanatory style. A full rewrite could repair both
problems while replacing a simple decomposition with a strategy the student
finds harder to reproduce.

This example separates three failure modes:

| Failure mode                 | What is inconsistent?                                       | A useful diagnostic                                                   | Candidate response                                                         |
| ---------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Broken semantic dependencies | Later claims or actions depend on a changed earlier result  | Recompute dependent claims; replay changed actions                    | Regenerate the affected continuation                                       |
| Local splice mismatch        | The inserted span differs from its surrounding trace        | Compare equivalent repairs with matched local structure               | Preserve surrounding notation, granularity, and tone                       |
| Global student mismatch      | The whole trace uses patterns poorly matched to the learner | Compare student likelihood, rollout behavior, and downstream learning | Use student prefixes, change scaffolding, or adjust the learning objective |

The diagnostics are imperfect. Low student likelihood can flag either a
useful new skill or an inaccessible target. High likelihood can reward a
familiar mistake. A sudden likelihood change around an edit boundary can also
reflect a necessary mathematical correction. **Compatibility scores should
help rank verified candidates; they should not replace correctness checks.**

“Same training data” is a helpful experimental control, but it does not make
the large and small models' conditional policies identical. Capacity,
optimization, and the RL stage can all change which strategies each model
can execute. Style is one possible contributor, not an established diagnosis
of every failed distillation run.

## Where a new study could fit

The space already contains teacher correction, style alignment, backtracking,
step-level supervision, selective masking, and student-aware teaching. None of
those phrases alone is a convincing novelty claim.

A narrower research question is:

> Given a fixed collection of RL-teacher trajectories and a fixed student,
> how should we choose the scope of a verified edit to improve downstream
> learning while removing a specified unwanted behavior?

I would study three questions in order:

1. **When does an edit hurt?** Separate broken dependencies, local splice
   effects, and mismatch between the original teacher and student.
2. **How far should a repair extend?** Compare a local patch, a regenerated
   suffix, a full rewrite, and discarding or resampling the example.
3. **Can the choice be predicted before expensive training?** Test whether
   dependency information, student compatibility, and continuation probes
   predict which strategy produces better held-out students.

The potential contribution is a measured decision rule over edit scope,
supported by controlled interventions. It is not yet a claim that such a rule
exists or is novel in every relevant subfield. If a fixed suffix-repair policy
matches a complicated selector, that is a useful result too.

This extends two earlier posts:
[A Repair Can Save a Trajectory Without Teaching the Agent](/posts/2026-08-14-counterfactual-trajectory-repair-sft-vs-rl/)
and
[What Makes an SFT Example Worth Learning?](/posts/2026-09-27-sft-quality-novelty-learnability/).
The former contains a small synthetic policy experiment, not an LLM training
result. Its distinction between immediate rescue and learning utility is
exactly why I would resist selecting edits only by how well they rescue a
frozen student's continuation.

## A controlled first experiment

Start with one domain that has a reasonably strong verifier, such as
executable coding tasks or mathematical problems with independently checked
answers. Choose one precisely observable target behavior. “Bad reasoning” is
too broad; “repeats an equivalent verification without new evidence” or
“returns code that mutates a forbidden input” is a testable specification.
The former needs an audited behavioral judge; the latter can often use tests.

Freeze the teacher traces, student checkpoint, task split, editor model, and
editor access to reference solutions. Any privileged information given to one
repair arm must also be available to the relevant controls. Keep task families
and near-duplicates out of both training and evaluation splits.

The first stage should vary the data transformation while keeping the student
update as ordinary hard-label SFT:

| Arm                           | Transformation                                                                                     | What it tests                                                                       |
| ----------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| A0: original                  | Train on the unchanged teacher pool                                                                | The value already present in the source data                                        |
| A1: filter / resample         | Reject unacceptable traces; seek a new acceptable trace for the same problem within a fixed budget | Whether editing beats a simpler acquisition policy                                  |
| A2: local patch               | Repair a span and retain the original suffix                                                       | Whether a narrow edit remains coherent; invalid candidates are rejected and counted |
| A3: suffix repair             | Keep a verified prefix and regenerate everything after the repair boundary                         | The value of restoring downstream dependencies                                      |
| A4: full rewrite              | Regenerate the whole demonstration                                                                 | Maximum freedom to repair, with greater representation drift                        |
| A5: constrained suffix repair | Use the A3 boundary while retaining the original trace's notation and step granularity             | Whether preserving local continuity helps beyond suffix repair                      |
| A6: style-only control        | Change presentation while preserving the audited semantic plan                                     | The effect of representation change without logical correction                      |

A0 is the observational baseline and can contain the original undesirable
behavior. The acceptance rules for the other arms must be explicit. In
particular, rejecting invalid A2 edits can change its problem coverage. Report
that change rather than silently comparing its easier surviving subset with
another arm's complete dataset.

Run two budget views. An **equal-problem comparison** asks what each method
does for the same input tasks, with its actual generation and training cost
reported. An **equal-compute comparison** allocates the same overall budget
and permits different numbers of accepted examples. These answer different
questions. We cannot generally hold problem coverage, accepted examples,
supervised tokens, and compute all constant when methods have different
acceptance rates and output lengths.

### Isolate style before attributing the result to style

Within an audited subset, cross semantic repair with presentation change:

|                                 | Preserve the source presentation               | Change presentation                   |
| ------------------------------- | ---------------------------------------------- | ------------------------------------- |
| Keep the original semantic plan | Original/control trace                         | Presentation-only rewrite             |
| Repair the semantic plan        | Repair with constrained notation and structure | The same repair expressed differently |

Match the two repaired targets on calculations, logical dependencies,
intermediate information, and approximate length. If one version adds a new
lemma or removes a necessary explanation, the contrast is not purely stylistic.
Call it a combined representation intervention instead.

Preserving the original **teacher** presentation should be one contrast.
Adapting to a measured **student-native** presentation should be a separate
contrast. Combining them would obscure whether the benefit comes from local
continuity or global student compatibility.

### Then test whether the missing ingredient is student state coverage

Only after the fixed-teacher comparison should the experiment change rollout
source. From a common student checkpoint, compare continued static SFT with
student rejection sampling, expert correction of student prefixes, and
SOPD-style independent step targets. Add a logits-based OPD baseline if the
infrastructure exists. Add preference training or RL if the question requires
them and the budget supports a fair comparison.

[Step-DPO](https://arxiv.org/abs/2406.18629) and
[SuperCorrect](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0967d7c8b171dd81b77c43067c02bebf-Abstract-Conference.html)
are especially relevant when the objective is to prefer a better decision
under the same prefix.
[LUFFY](https://arxiv.org/abs/2504.14945) is relevant when teacher demonstrations
need to be combined with student exploration under RL. They belong in a later
comparison because changing the update objective in the first experiment
would confound the effect of editing.

## Preparing correction data without silently changing the task

There are two distinct training targets: preventing the original error and
recovering after it has occurred. For prevention, construct the continuation
from a valid prefix before the error. For recovery, retain the actual erroneous
prefix and train an explicit correction that makes sense given that history.
A recovery target may need to retract a claim before proceeding.

Masking an erroneous token's label removes its direct positive supervision.
It does not remove the token from the context. Conversely, deleting the error
from the input can destroy the recovery task. Preserve the distinction in the
data format and inspect the actual shifted labels the trainer consumes.

I would make the preparation pipeline produce an auditable record for every
candidate:

1. **Source and diagnosis.** Store the prompt, original trace, model versions,
   first suspected error, target behavior, and affected dependencies.
2. **Repair proposal.** Save the unmodified prefix, edit boundary, candidate
   continuation, editing prompt, and access to reference answers.
3. **Independent acceptance.** Check final correctness, internal consistency,
   and the specified behavior separately. Blind the judge to method names where
   possible, and audit a sample manually.
4. **Execution validity.** For agents, replay changed actions and record real
   observations. Reject traces whose purported environment state cannot be
   reproduced.
5. **Training representation.** Preserve role boundaries, stop tokens, label
   masks, token counts, and truncation status. Inspect the packed examples for
   unintended attention between independent targets.
6. **Selection accounting.** Record accepted and rejected candidates, sampling
   attempts, retained task coverage, and construction cost. Keep provenance
   through deduplication and train/evaluation splitting.

The editor should receive an explicit scope: diagnose first, identify the last
verified prefix, and regenerate the dependent continuation. “Make this better”
is too unconstrained for a controlled comparison. A useful initial prompt
specifies the behavior to fix, the claims that must remain true, the allowed
edit boundary, the presentation constraints, and the required output fields.
Its output still needs independent verification.

For the first implementation, the following routing policy is a **proposed
heuristic**, not a result established by the survey:

| Choice            | Initial condition for trying it                                                            |
| ----------------- | ------------------------------------------------------------------------------------------ |
| Keep              | The trace passes the specified correctness and behavior checks                             |
| Local edit        | The defect is isolated and downstream claims or observations remain valid after rechecking |
| Regenerate suffix | A valid prefix exists, but the edit changes later dependencies                             |
| Full rewrite      | No reliable prefix remains, or several interacting defects defeat bounded repair           |
| Drop / resample   | No candidate passes verification within the allocated budget                               |

Generate multiple verified candidates only where the expected information gain
justifies the cost. Record student likelihood and continuation success as
features, then test whether they predict post-SFT gains on development tasks.
Do not train the router to maximize those proxies and assume that the learning
problem has been solved. The router must ultimately earn its complexity by
beating a fixed repair policy on held-out student evaluations.

## Evaluate the student, not just the repaired artifact

A repaired trace passing a verifier measures data quality. A frozen student
successfully continuing after a patch measures immediate compatibility or
rescue. The research outcome is the behavior of the independently evaluated
student **after training**.

Let the main joint metric be

$$
J = P(\text{correct result} \land \text{no specified unwanted behavior}).
$$

Report accuracy and behavior frequency separately as well. A model that stops
attempting difficult tasks might reduce bad-behavior frequency while becoming
less useful. Report behavior both per task and, where meaningful, per audited
opportunity to exhibit it. Include an unrelated retention set, output length,
repetition, truncation, and total data-construction plus training cost.

Use held-out task families and multiple training seeds for the final
comparison. Match decoding settings and report uncertainty across both tasks
and seeds. If many candidate selectors are explored, choose them on development
data and reserve a final test set. Do not claim a universal negative result
from a confidence interval that still admits a practically useful gain.

A manageable proposed pilot is to audit several hundred traces, test candidate
repairs with a frozen student, and run a small single-seed SFT screen. Then
confirm only the most informative arms with at least three training seeds and
a second student size. These are planning choices, not a power calculation;
the pilot's variance should determine the final sample size. No GPU-hour
estimate is credible until sequence lengths, model sizes, and hardware are
specified.

The failure cases would be informative. If style controls do not matter after
dependency repair, the splice explanation weakens. If only student-prefix
methods help, coverage is likely the more productive direction. If full
rewriting wins at the same total cost, a complicated local editor needs a
stronger justification. If repair quality improves while post-SFT behavior
does not, the acceptance proxy has failed to predict learning utility.

My first implementation would therefore be **verified suffix repair with
explicit presentation constraints**, compared against unchanged traces,
resampling, local edits, and full rewrites. It is a practical starting
hypothesis because it addresses dependent errors without immediately requiring
an online training stack. The study should determine when that choice helps,
when it should be replaced, and whether its apparent benefit survives a fair
comparison of coverage and cost.
