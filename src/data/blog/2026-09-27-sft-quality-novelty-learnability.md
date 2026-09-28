---
author: Jing Lu
pubDatetime: 2026-09-27T17:00:00-07:00
modDatetime: 2026-09-27T17:32:00-07:00
title: "What Makes an SFT Example Worth Learning? Quality, Novelty, and Learnability"
featured: true
draft: false
tags:
  - AI
  - Post Training
  - Reinforcement Learning
  - Evaluation
  - ML Engineering
description: "From curriculum learning and DAgger to LIMA, LESS, and on-policy distillation: a source-backed framework for separating SFT data quality, novelty, and learnability, with executable examples and a small negative result."
---

A correct answer is not necessarily a useful training example. An unfamiliar
answer is not necessarily a new skill. And an answer that a model can memorize
is not necessarily one it can learn to use elsewhere.

These distinctions matter when choosing data for supervised fine-tuning, or
SFT. Suppose we already have a fixed collection of offline demonstrations. We
can run the current model over those demonstrations, inspect its probabilities,
and change how much each example contributes to training. Can that make better
use of supervision without generating another training rollout?

Before choosing a weighting function, I want to define three quantities:
**quality**, **novelty**, and **learnability**. This post gives working
definitions, concrete measurements, and an experiment that can challenge the
idea rather than assume it works.

The evidence has three different levels. The code examples below are
handwritten and executable. A small synthetic policy experiment has actually
run. The Qwen Coder experiment is a proposed next step; there are no Qwen
training results in this post.

## How we got here: three questions that became entangled

This problem sits at the intersection of several research traditions. They
address different limitations of learning from demonstrations; putting them
together explains why one scalar score is unlikely to settle the question.

The first question is **when to present an example**.
[Bengio et al.'s Curriculum Learning (2009)](https://icml.cc/2009/papers/119.pdf)
studied training schedules that introduce examples in a meaningful progression.
It made example order part of the optimization problem rather than treating
the training set as an unordered collection.
[Kumar, Packer, and Koller's self-paced learning (2010)](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html)
then let the current model help determine which examples were easy enough to
include, expanding the selected set over training. These results motivate
model-dependent selection. They do not establish that a fixed band of
language-model likelihoods is the right curriculum.

The second question is **which contexts the learner needs to handle**.
[Ross, Gordon, and Bagnell's DAgger (2011)](https://proceedings.mlr.press/v15/ross11a.html)
addressed a failure of ordinary imitation learning: the learner's actions
change the observations it encounters. Good predictions on expert states can
therefore coexist with poor behavior on the learner's own states. DAgger
iteratively collects states under a learner-involved policy, obtains expert
labels, and aggregates the data. For an autoregressive model, earlier
generated tokens similarly change later contexts. This connection explains
why a correct teacher trajectory may still be poorly matched to student
deployment; it does not make every offline trajectory unhelpful.

The third question is **which demonstrations deserve the training budget**.
[LIMA (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206) demonstrated strong
instruction-following results from 1,000 carefully curated examples on a
pretrained 65B model. It helped sharpen the question of selection rather than
volume alone. That finding should not be read as a guarantee that 1,000
examples can teach arbitrary missing capabilities to a much smaller student.
[DEITA (Liu et al., ICLR 2024)](https://arxiv.org/abs/2312.15685) made selection
more explicit by studying quality, complexity, and diversity. Those are useful
data dimensions, but dataset diversity and a particular student's missing
skills are still different objects.

On-policy distillation connects the context question back to supervision.
[GKD (Agarwal et al., ICLR 2024)](https://arxiv.org/abs/2306.13649) lets the
student produce trajectories and uses teacher feedback on those trajectories.
This changes where supervision is applied. With an immutable offline dataset,
we cannot reproduce that state-collection step simply by renaming a score
"on-policyness."

The narrower question in this post follows from that constraint: **among the
correct demonstrations we already possess, which ones help this student make
progress on the capabilities we care about?** Quality determines whether the
target is acceptable. Novelty describes what differs from existing behavior.
Learnability asks whether training can turn that difference into transferable
competence.

## Start with one problem and several answers

Consider this specification:

> Given a list of integers, return whether it contains a duplicate. Do not
> modify the input. For now, there is no explicit runtime threshold.

A model can solve it with a nested loop:

```python
def contains_duplicate(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False
```

It can use a set:

```python
def contains_duplicate(nums):
    seen = set()
    for value in nums:
        if value in seen:
            return True
        seen.add(value)
    return False
```

Or it can sort a copy and compare adjacent elements:

```python
def contains_duplicate(nums):
    ordered = sorted(nums)
    return any(
        ordered[i] == ordered[i - 1]
        for i in range(1, len(ordered))
    )
```

All three satisfy the functional specification. Now consider:

```python
return len(set(nums)) <= len(nums)
```

This expression always returns `True`. It looks compact and familiar, but it
fails on every input without duplicates.

In the [executable example](/data/offline-sft-frontier/coding_examples.py), the
three correct implementations pass all ten demonstration tests. The incorrect
one passes five. The tests include empty lists, single elements, negative
values, and repeated values. Ten tests are useful evidence about those cases;
they are not a proof of universal correctness.

Suppose a particular checkpoint already uses nested loops reliably but rarely
uses sets. Teaching the set-based solution might expand its usable strategies.
Renaming the loop variables probably would not. That statement depends on the
checkpoint: we must measure its behavior rather than assume that an algorithm
is new to it because it looks more advanced to us.

## Quality: does the demonstration meet the specification?

**Quality is the validity of a demonstration relative to a task specification,
a verifier, and the scope of evaluation.** With the specification fixed, it is
primarily a property of the task and answer, not of the student's confidence.

For executable code, a first measurement is:

$$
\widehat Q(s;\mathcal T)
=\frac{1}{|\mathcal T|}
\sum_{t\in\mathcal T}
\mathbf 1[\text{program in }s\text{ passes }t].
$$

This is the observed pass fraction on a named test suite. It is not
automatically a calibrated probability that the program is correct.

This limitation is observable, not just philosophical.
[EvalPlus (Liu et al., 2023)](https://arxiv.org/abs/2305.01210) expanded the
tests used to evaluate generated programs and exposed errors missed by the
original suites; it also found that test insufficiency could affect model
rankings. For our pipeline, a demonstration can be "verified" under one suite
and rejected under a stronger one. Verifier version and coverage therefore
belong in the sample's quality metadata.

Quality also has multiple dimensions. Functional correctness, input
preservation, runtime constraints, and agreement between an explanation and
its code should initially be recorded separately. A single blended score can
hide a failure in one dimension behind strength in another.

For the duplicate problem, the nested-loop answer is functionally correct.
If the specification adds a strict runtime requirement on large inputs, that
same answer may no longer qualify. We should not quietly change the task's
requirements when deciding which sample is better.

The first experiment can use quality as an eligibility gate: retain solutions
that pass the declared checks, then compare different correct strategies. This
keeps an experiment about supervision selection from becoming an experiment
about filtering obvious errors.

## Novelty: what does this add relative to the current model?

**Novelty is the additional content a demonstration provides relative to a
specified model's existing behavior and skill coverage.** It is not an
absolute label attached to the text.

There are at least three different things people mean by novelty:

| Kind                           | Possible measurement                                                    | Example                                                          | What it can miss                                         |
| ------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------- |
| Expression-level unfamiliarity | Token negative log-likelihood, centered log-likelihood, low-score spans | The set implementation receives lower likelihood than the loop   | Unusual names and incorrect code can also be unlikely    |
| Strategy difference            | Algorithm labels, normalized syntax trees, data flow, tool-use patterns | Set membership instead of comparing every pair                   | A structural difference need not add a useful capability |
| Capability gap                 | Failure rate on a defined skill evaluation                              | The model struggles on tasks requiring a remembered set of items | Task difficulty and missing skill can be confounded      |

Changing `seen` to `visited_values` changes tokens. Replacing a nested loop
with a set changes the method. Reliably transferring set-based reasoning to a
new problem changes what the model can do. These are related observations, but
they are not interchangeable measurements.

If archived baseline outputs are available, strategy frequency is another
candidate feature. For example, imagine 90 loop solutions and ten set solutions
among 100 valid outputs on a fixed task family. The set strategy is less
frequent under that sampling setup. Those numbers are illustrative, not an
observed result here. Frequency estimates need uncertainty and decoding
metadata; zero observations do not prove zero probability.

If new rollouts are forbidden and no historical outputs exist, offline scoring
can still measure conditional likelihood. It cannot fully identify the
model's actual strategy coverage. Dataset diversity is also not the same as
model-relative novelty.

For an initial study, I would store expression-level scores and explicit
algorithm labels separately. Then I would ask whether either predicts
independent learning gains. Calling a score "novelty" should not substitute
for validating what it measures.

There is an older reason to separate surprise from progress.
[Oudeyer, Kaplan, and Hafner (2007)](https://www.pyoudeyer.com/ims.pdf) studied
intrinsic motivation based on learning progress. Their approach directs
exploration toward situations where prediction can improve, rather than
rewarding every large prediction error. Already predictable situations offer
little progress, while irreducibly unpredictable ones can remain surprising
without becoming more learnable. This is an intellectual precedent for a
moving learning frontier, not direct evidence for our CLL weighting rule.
In code, arbitrary identifier changes provide a mundane analogue: surprise
can increase while algorithmic content stays fixed.

## Learnability: can this model absorb the supervision within a budget?

**Learnability is the model's ability to acquire and transfer a skill under a
specified training procedure and budget.** It depends on the initial
checkpoint, the supervision, the optimizer, and the amount of training.

A practical measurement for a group of demonstrations $S$ is:

$$
\widehat L_B(S;\theta,\mathcal A)
=\operatorname{Score}_{V_{\mathrm{skill}}}
\left(\mathcal A_B(\theta,S)\right)
-\operatorname{Score}_{V_{\mathrm{skill}}}(\theta).
$$

Here $\mathcal A_B$ is a declared training procedure with budget $B$, and
$V_{\mathrm{skill}}$ contains independent examples of the relevant skill. The
quantity can be negative. It measures the result of an intervention on a
sample group, not an intrinsic property of one sentence.

Suppose the model initially passes 30 of 100 held-out set-related problems.
After training, it passes 55. A matched control trained for the same budget
passes 38. The raw gain is 25 percentage points; the extra gain over the
control is 17 points. These are illustrative arithmetic, not measurements
from the current experiment.

The control matters. Without it, we cannot distinguish the value of these
demonstrations from improvement that another equally sized training set would
have produced. Multiple seeds, task-level splits, and checks for regression on
other skills are also necessary.

A lower loss on the training answer measures improved fit. To test learning,
we need fresh problems that exercise the same underlying skill. Merely
renaming variables is a weak transfer test. New input formats, different
objects to remember, and different uses of membership can probe more of the
capability.

CLL may eventually help predict learnability before training. It does not
measure learnability directly. The prediction target still needs an actual
training intervention and an independent evaluation.

[Dataset Cartography (Swayamdipta et al., 2020)](https://aclanthology.org/2020.emnlp-main.746/)
offers a practical bridge between a static score and a full intervention. It
tracks confidence in the target label and its variability across training
epochs. In the classification datasets studied, ambiguous examples were
useful for OOD generalization, easy examples supported optimization, and
hard-to-learn regions often contained labeling errors. The relevant lesson is
to inspect trajectories of learning, not equate one high loss with useful
difficulty. Its ambiguity measure is not the same as middle-range CLL, and
its findings do not establish an inverted-U law for generative SFT.

## Learning value is the target, not a presumed product

The expression

```text
quality × novelty × learnability
```

is a useful reminder to consider all three. It is not yet a justified
multiplicative law. The quantities have different units, are not independent,
and may already contain overlapping information.

A more explicit target is incremental utility under a fixed budget:

$$
U_B(S)=
\frac{
\operatorname{Score}_{V_{\mathrm{target}}}(\theta_{S,B})
-\operatorname{Score}_{V_{\mathrm{target}}}(\theta_{\mathrm{control},B})
}{\operatorname{Cost}(S,B)}.
$$

The target evaluation can be broader than the skill evaluation used for
learnability. A model might acquire set usage while losing performance
elsewhere. Cost can be measured in training tokens or GPU seconds, but the
comparison needs a consistent definition. This is an evaluation objective,
not a score we can obtain for free from one forward pass.

Several outcomes remain plausible. Familiar, correct examples can reinforce
useful behavior or prevent forgetting. Unfamiliar, correct examples can add a
strategy. Unfamiliar, incorrect examples can teach errors. Correct examples
that the model cannot absorb within the available budget may require a
different presentation, decomposition, or training method.

The experiment should distinguish these possibilities rather than assume
that "more novel" always means "more valuable."

One related method already asks a more targeted question than "is this
example difficult?" [LESS (Xia et al., 2024)](https://arxiv.org/abs/2402.04333)
uses an optimizer-aware influence approximation and low-dimensional gradient
features to select instruction data aligned with a few examples of a desired
capability. It makes the target task part of data selection.

The connection can be seen in a local SGD calculation. Let $g_s$ be the
training-loss gradient of a candidate and $g_V$ the loss gradient on a
separate development set. For one sufficiently small update:

$$
L_V(\theta-\eta g_s)
\approx L_V(\theta)-\eta\,g_V^\top g_s.
$$

A large candidate loss does not determine the sign of this alignment. This
first-order illustration is not the full LESS algorithm; Adam, multiple
updates, and feature storage require additional treatment. It suggests a
useful next comparison: does a cheap CLL feature predict actual gains as well
as a more expensive, target-aware gradient signal? Any such development set
must remain separate from the final test set.

## Where on-policy data and CLL fit

Strictly speaking, on-policy describes a sampling relationship. A response is
sampled from the current policy. Re-scoring a fixed offline response does not
change the policy that originally generated it.

In on-policy distillation, the student can generate the trajectory while a
teacher supplies supervision at the visited contexts. This is the distinction
used in [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649).
The origin of the contexts and the origin of the supervision need not be the
same.

For offline data, we can instead study compatibility with the current model.
One feature is centered log-likelihood, or CLL:

$$
z(y,c)=\log p(y\mid c)+H[p(\cdot\mid c)].
$$

It compares a token's log-probability with the model's own average
log-probability at that context. Under $y\sim p$, its conditional expectation
is zero. Under another distribution $q$, however, the identity is:

$$
\mathbb E_q[z]=H(p)-H(q)-D_{\mathrm{KL}}(q\|p).
$$

This matters: an external distribution does not necessarily have negative
mean CLL. If $p$ is uniform, every token has CLL zero, regardless of the
distribution that produced the observed token. CLL is not a KL distance or a
posterior probability of on-policy origin.

[DDT and IDFT](https://arxiv.org/abs/2602.12222) study CLL and its use in
fine-tuning. The identity above is a separate algebraic check on how we use
the feature here. An empirical scoring method can still be useful without
being a universal distribution detector.

There is another distinction for agents: a plausible next action does not
imply a reachable context. Consider these hand-specified paths in a
deterministic environment:

| Path                      | First action probability | Read probability given the path | Calculator probability given the path | Probability of the prefix before Calculator |
| ------------------------- | -----------------------: | ------------------------------: | ------------------------------------: | ------------------------------------------: |
| Common search route       |                     0.80 |                            0.80 |                                  0.90 |                                        0.64 |
| Rare archive-search route |                     0.02 |                            0.80 |                                  0.90 |                                       0.016 |

If the final action distribution is $[0.9,0.1]$ in both contexts, its local
CLL is identical. Yet the prefix probabilities differ by a factor of 40.
This calculation treats the full history as the state. With aggregated states
or stochastic tool observations, state visitation requires more information.
Tool outputs are environment observations, not model actions.

## What the recent SFT methods change, and what remains open

These distinctions also help separate several methods that can otherwise
sound like versions of the same idea.

[Dynamic Fine-Tuning, or DFT (Wu et al., 2025; ICLR 2026)](https://arxiv.org/abs/2508.05629)
rescales token learning with the model's own token probability. It provides
a direct probability-based reweighting baseline. The frozen probability
weights in our toy experiment below are not a faithful DFT reproduction:
the original method updates its weights with the model during training.

IDFT uses CLL-related loss modulation; the same paper's Hinted Decoding
changes how responses are produced. Those are different interventions.
PEAR uses importance-style offline loss reweighting at several granularities
and evaluates whether the resulting SFT checkpoint is a better starting point
for RL. Our CLL band is neither an importance ratio nor a reproduction of
either objective.

| Research direction                 | What it changes                                         | Question it helps answer                                         | What is still needed here                                       |
| ---------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------- |
| Curriculum and self-paced learning | Order or inclusion as learning proceeds                 | When should a learner see an example?                            | A validated notion of progress for this model and task          |
| DAgger and GKD                     | Visited contexts and where expert feedback is provided  | Is supervision applied where the learner actually goes?          | New state collection, which fixed offline data cannot supply    |
| DEITA                              | Selection using quality, complexity, and diversity      | Which subset uses a limited alignment budget well?               | Separating dataset diversity from student-relative novelty      |
| Dataset Cartography                | Analysis of behavior across training                    | Which examples are stable, ambiguous, or persistently difficult? | Testing whether these signals predict generative skill transfer |
| LESS                               | Target-conditioned selection using gradient information | Which examples are likely to help the chosen capability?         | Cost comparisons and independent confirmation of gains          |
| DFT, IDFT, and PEAR                | Different forms of offline loss reweighting             | Can supervision allocation improve generalization or later RL?   | Testing whether a middle band helps beyond existing baselines   |

This is a map of related questions, not a claim that the methods share one
derivation. The possible contribution of a frontier study would be an
empirically validated relationship between model-relative supervision and
learning gains, rather than a new name for data selection.

## A concrete weighting rule, with a limited claim

We can test whether moderately unfamiliar supervision is especially useful.
One candidate assigns a weight to the mean CLL $z_k$ of a complete reasoning
step or code block:

$$
w_k=\epsilon+(1-\epsilon)
\exp\left[-\frac{(z_k-m)^2}{2\sigma^2}\right].
$$

The center $m$ and width $\sigma$ specify the emphasized band. The floor
$\epsilon$ keeps other eligible examples from receiving zero weight. These
parameters must be selected without using the final test set.

For a hand-calculable illustration, treat the three duplicate-detection
algorithms as three semantic actions with probabilities $[0.8,0.15,0.05]$.
Their entropy is approximately $0.613$ nats. With $m=-1.3$, $\sigma=0.6$,
and $\epsilon=0.1$:

| Semantic action  | Probability |    CLL | Raw weight |
| ---------------- | ----------: | -----: | ---------: |
| Nested loop      |        0.80 |  0.390 |      0.117 |
| Hash set         |        0.15 | -1.284 |      1.000 |
| Sort and compare |        0.05 | -2.383 |      0.277 |

These are invented semantic-action probabilities, not measured Qwen token
scores. The table shows what the rule does. It does not establish that the
set solution has the highest learning value.

The weighted loss can use the same weight throughout a complete step:

$$
\mathcal L=
-\frac{
\sum_k\operatorname{sg}(w_k)
\sum_{t\in k}\log\pi_\theta(y_t\mid x,y_{<t})
}{\sum_k\operatorname{sg}(w_k)|k|}.
$$

The stop-gradient operator $\operatorname{sg}$ makes weights fixed
coefficients for the update. Normalization occurs over the whole optimization
batch. A single sequence weight divided by itself in a one-example batch has
no effect. Gradient accumulation and distributed training must preserve the
intended batch-level denominator.

Normalization also does not remove every optimization confound. Weighting
changes gradient directions and effective sample size. Comparisons should
include checks for effective update strength rather than attributing every
gain to a better curriculum.

## What the small experiment actually found

I implemented a three-parameter binary policy:

$$
\pi(a=+1\mid x)=\operatorname{sigmoid}(w^\top x).
$$

Correct labels come from
$\operatorname{sign}([0.8,1.2,-0.7]^\top x)$, while the initial policy uses
$w=[1.6,0,0]$. The model therefore has a specific initial limitation: it relies
only on the first feature.

Each of five seeds has 1,000 fixed correct demonstrations. Each method starts
from the same parameters and receives 200 updates with batches of 32. Scores
and weights are frozen at initialization. An independent OOD evaluation
changes feature standard deviations from $[1,1,1]$ to $[0.5,1.5,1.5]$ while
keeping the labeling rule fixed.

| Method                       | OOD accuracy after SFT | OOD expected reward after SFT |
| ---------------------------- | ---------------------: | ----------------------------: |
| Uniform SFT                  |         99.12% ± 0.92% |                88.27% ± 0.68% |
| Frozen probability weighting |         95.58% ± 1.02% |                85.41% ± 0.88% |
| Monotone CLL weighting       |         96.75% ± 0.88% |                86.49% ± 0.82% |
| CLL band weighting           |         98.82% ± 1.03% |                88.48% ± 0.62% |

The values are means and sample standard deviations across seeds, not
confidence intervals. Accuracy measures the greedy decision; expected reward
measures the probability of a correct sampled action.

The band-weighted model has slightly higher expected reward and slightly
lower accuracy than uniform SFT. This is not clear evidence of an advantage.
After an equal number of exact expected-reward optimization steps, OOD expected
reward is 90.49% for uniform SFT and 90.55% for band weighting. That stage
enumerates the two actions analytically; it is not sampled PPO, GRPO, or an
LLM RL experiment.

Five additional runs per seed train on individual CLL quintiles. The second
quintile does well on some metrics, but feature distributions differ between
quintiles, and each contains only 200 unique examples. These comparisons are
exploratory, not causal evidence for an optimal novelty band.

There is also a useful scoring trap. In this binary setting, CLL for a
correct action with probability $p$ is
$(1-p)\log[p/(1-p)]$. It is zero at $p=0.5$ and approaches zero as $p$ approaches
one. High CLL is not a globally monotonic ranking of confidence. Naming the
quintiles "easy" through "hard" would misdescribe them.

The [raw results](/data/offline-sft-frontier/results.json) and
[reproduction instructions](/data/offline-sft-frontier/README.md) are available.
The experiment uses only the Python standard library. It is a measurement
check, not evidence that this method improves large language models.

## The next experiment: small Qwen Coder models

For a low-cost starting point, I would use
[Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
to check the implementation and
[Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
for the first comparisons. These are deliberately small, older models. The
purpose is to spend the budget on controls and repetition before scaling.

The proposed initial dataset has 200–500 short code tasks, each with two or
three verified solutions and an algorithm label. Splits are made by problem
and task family, not by answer: different solutions to the same problem must
not leak across train and evaluation splits.

The first comparison has three methods: uniform SFT, monotone CLL weighting,
and band weighting. It freezes the scoring checkpoint and matches training
token exposure. It evaluates independent problems from trained skill families
and withheld families, with additional controls for quality, length, and update
strength. Short-context LoRA is a candidate setup; memory use and throughput
need a smoke test before choosing a larger budget.

Two practical resource options are
[Colab's free compute](https://research.google.com/colaboratory/faq.html), whose
availability is not guaranteed, and
[Modal's Starter credits](https://modal.com/pricing). At the September 27,
2026 check, Modal listed a monthly free credit of 30 US dollars. Dedicated
hourly GPUs are another option when repeated runs need a predictable setup;
prices should be checked when scheduling the experiment.

A cheap chat API is not necessarily sufficient. Exact entropy requires the
full vocabulary distribution; a few returned top-token probabilities do not
provide it. Custom weighted training also needs access to the backward pass.

Only after a repeatable SFT difference would I add dynamic rescoring and
actual downstream RL. [PEAR](https://arxiv.org/abs/2602.01058) motivates this
last evaluation: in its experiments, a stronger SFT checkpoint did not always
produce a stronger model after the same RL training. Immediate SFT quality
and the quality of an RL initialization are different outcomes.

The literature changes what should count as a convincing follow-up. A
same-size random subset helps check whether selection adds value at all.
Quality-only and diversity-aware selection help check whether the apparent
frontier is just a better dataset. Dynamic probability weighting distinguishes
a frozen-score experiment from a real DFT baseline. A small target-aware
gradient comparison tests whether likelihood is leaving useful information
out. These can be staged as the budget permits; none has been established by
the current toy result.

## The question worth keeping open

The hypothesis is that useful supervision may occupy a moving region between
what a model already does reliably and what it cannot yet absorb. A
band-shaped weighting function is one way to test that hypothesis. It is not
the hypothesis's proof.

The research questions are concrete:

1. After controlling quality, which features available before training predict
   gains on independent problems?
2. How strongly does expression-level unfamiliarity track a change in algorithm
   or skill coverage?
3. Is there a useful intermediate region, and does its location change with
   the checkpoint and training budget?

A monotonic relationship, a weak relationship, or no stable relationship are
all possible answers. Good measurements should let us discover any of them.

## References and a suggested reading path

The papers below support different parts of the argument. The working
definitions and proposed CLL-band experiment are this post's synthesis, not a
framework validated jointly by these sources.

1. **Oudeyer, Kaplan, and Hafner (2007).** [Intrinsic Motivation Systems for Autonomous Mental Development](https://www.pyoudeyer.com/ims.pdf). The distinction between persistent surprise and opportunities for learning progress.
2. **Bengio, Louradour, Collobert, and Weston (2009).** [Curriculum Learning](https://icml.cc/2009/papers/119.pdf). Why presentation order can matter to optimization and generalization.
3. **Kumar, Packer, and Koller (2010).** [Self-Paced Learning for Latent Variable Models](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html). A model-dependent mechanism for expanding the selected training set.
4. **Ross, Gordon, and Bagnell (2011).** [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html). DAgger and the consequences of learner-induced state distributions.
5. **Swayamdipta et al. (2020).** [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746/). Confidence and variability as diagnostics of example behavior during training.
6. **Zhou et al. (2023).** [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206). A demonstration of the value of carefully curated instruction data.
7. **Jiawei Liu et al. (2023).** [Is Your Code Generated by ChatGPT Really Correct?](https://arxiv.org/abs/2305.01210). EvalPlus and the limits of treating a weak test suite as a correctness oracle.
8. **Wei Liu et al. (ICLR 2024; preprint 2023).** [What Makes Good Data for Alignment?](https://arxiv.org/abs/2312.15685). DEITA's separation of quality, complexity, and diversity.
9. **Xia et al. (2024).** [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333). Selection informed by the capability we want to improve.
10. **Agarwal et al. (ICLR 2024; preprint 2023).** [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649). GKD and teacher supervision on student-generated trajectories.
11. **Wu et al. (ICLR 2026; preprint 2025).** [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](https://arxiv.org/abs/2508.05629). DFT's dynamic probability-based token weighting.
12. **Miaosen Zhang et al. (2026).** [Towards On-Policy SFT: Distribution Discriminant Theory and its Applications in LLM Training](https://arxiv.org/abs/2602.12222). DDT, IDFT, and Hinted Decoding; the closest context for the CLL feature considered here.
13. **Dylan Zhang et al. (2026).** [Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning](https://arxiv.org/abs/2602.01058). PEAR and downstream RL as an evaluation target for SFT.
