---
author: Jing Lu
pubDatetime: 2026-09-27T17:00:00-07:00
title: "What Makes an SFT Example Worth Learning? Quality, Novelty, and Learnability"
featured: true
draft: false
tags:
  - AI
  - Post Training
  - Reinforcement Learning
  - Evaluation
  - ML Engineering
description: "Working definitions, executable examples, and a small negative result for selecting offline SFT data: distinguish correctness, unfamiliarity, and actual learning gains before designing a frontier-weighted loss."
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
