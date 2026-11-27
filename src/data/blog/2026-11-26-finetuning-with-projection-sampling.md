---
author: Jing Lu
pubDatetime: 2026-11-26T22:05:00-08:00
title: "What Projection Sampling Changes About SFT"
featured: false
draft: false
tags:
  - AI
  - LLM
  - Post Training
  - Reinforcement Learning
  - Evaluation
  - ML Engineering
description: "An analysis of Finetuning with Sampling that compares projection sampling, expert SFT, on-policy distillation, and RL, with a runnable example of the sampling shortcut."
---

A small model is learning to solve algebra problems. Given $x+1/x=3$, it must find $x^2+1/x^2$. One expert solution invokes a recurrence for symmetric powers. Another squares the given expression and subtracts the cross term, giving $3^2-2=7$. Both solutions are correct and give the student different sequences of decisions to imitate.

Now imagine the expert consistently uses methods the student rarely produces. Supervised fine-tuning (SFT) has a direct training signal for every expert token. Reinforcement learning (RL) can train on the student's own attempts, though it needs successful exploration or a reward that distinguishes useful partial attempts. Aayush Karan, Sitan Chen, and Yilun Du ask whether an expert solution can guide the search for a training target that the student is already more likely to produce. Their method, **projection sampling**, does the search before fine-tuning and uses ordinary SFT afterward. [Finetuning with Sampling](https://arxiv.org/abs/2610.02140)

I previously looked at how to repair individual errors in teacher traces. Projection sampling raises a different question. Given several valid solutions, how should we allocate training probability among them? The answer is conditional on the learner, and the difference between the mathematical answer and the paper's implementation matters. [Earlier post on trace repair](https://ajing.github.io/posts/2026-10-01-repairing-reasoning-traces-before-distillation/)

## Table of contents

## Which correct solutions should survive?

Fix a problem and let $p$ be the unassisted base model's distribution over answers. Let $C$ contain the solutions that preserve the information we want to teach. For our algebra example, $C$ might require a correct justification and the answer 7. A final-answer checker implements a weaker constraint. It would accept a trace that makes a cancellation error and happens to end with 7. The definition of $C$ determines how much of the expert's work survives. [Paper §4.1](https://arxiv.org/html/2610.02140v1#S4.SS1)

Among distributions supported on $C$, the one closest to $p$ in $D_{\mathrm{KL}}(q\Vert p)$ is $p$ conditioned on $C$:

$$
p_C(y)=\frac{p(y)\,\mathbf{1}[y\in C]}{p(C)}.
$$

This assumes $p(C)>0$. The projection preserves the base model's relative probabilities within the valid set. If it assigns twice as much probability to one valid solution as another, their projected probabilities still have a ratio of two to one. That makes it a distribution over ways to solve the problem, with room for less common valid answers. [Paper §4.1, Proposition 1](https://arxiv.org/html/2610.02140v1#S4.SS1)

With a reliable verifier, ordinary rejection sampling can target that distribution by drawing from $p$ and keeping answers in $C$. At a 1% success rate, getting one accepted answer takes 100 generations in expectation. This gives a concrete reason to use a worked solution. It provides an informed starting point and helps the model propose new answers where unrestricted sampling spends most of its time failing. An independent best-of-$n$ rewrite is another natural baseline because it uses the expert hint without maintaining a chain. [Paper §4.1–4.3](https://arxiv.org/html/2610.02140v1#S4)

## Exact sampling and the shortcut

The theoretical sampler proposes a change from $x$ to $y$ with probability $k(y\mid x)$ and accepts it with probability

$$
\min\left(1,\frac{p_C(y)k(x\mid y)}{p_C(x)k(y\mid x)}\right).
$$

The reverse proposal term corrects for asymmetry. A proposal can be easy to generate in one direction and hard in the other. Under the stated Markov-chain conditions, Metropolis–Hastings has $p_C$ as its stationary distribution. Its distribution-level KL contraction is compatible with occasionally accepting a less likely answer. [Paper §4.2](https://arxiv.org/html/2610.02140v1#S4.SS2)

In the language-model implementation, the sampler works in blocks. It keeps a prefix, chooses a random cut within the current sequence, and generates a replacement suffix with the expert solution in context. For the experiments, the authors save the cost of scoring reverse proposals by replacing the current candidate whenever the new one has higher base-model likelihood. That is a hill-climbing acceptance rule. The expert-conditioned proposal encourages information preservation, and the reported math dataset still has 95.33% accuracy after sampling. An implementation that uses this shortcut can concentrate on a few highly likely solutions. [Paper §4.3 and Appendix C](https://arxiv.org/html/2610.02140v1#A3)

A four-state example shows what this choice changes. The base model assigns probabilities 0.24, 0.18, and 0.12 to three valid solutions and 0.46 to an invalid one. The projected probabilities over the valid trio are 44.44%, 33.33%, and 22.22%. Starting at the least likely valid solution, exact MH and greedy replacement receive the same asymmetric proposal distribution. The calculation propagates probability mass exactly for both rules. KL is measured in nats.

| Update steps | Exact MH KL to projection | Greedy KL to projection |
| -----------: | ------------------------: | ----------------------: |
|            0 |                    1.5041 |                  1.5041 |
|            1 |                    0.0856 |                  0.0450 |
|            2 |                    0.0065 |                  0.0617 |
|           12 |              Under 0.0001 |                  0.7419 |

At step 12, greedy has raised the expected base-model log probability and put 98.62% of its mass on one solution. Its distance from the projection has grown since step 1. Exact MH converges toward the original 44:33:22 allocation. This small calculation isolates the acceptance rules. The language-model training results below come from the paper. For a real data pipeline, track likelihood together with validity, solution diversity, and downstream performance. [The exact target and the experimental shortcut](https://arxiv.org/html/2610.02140v1#A3.SS2)

[Download the Python experiment](/data/finetuning-with-sampling/projection_sampling.py) and run it with Python 3.9 or later. It uses the standard library.

```bash
python3 projection_sampling.py --steps 12
```

The paper estimates the number of generated tokens for data construction as roughly $|\mathcal{D}|N_{\mathrm{MCMC}}T^2/(4B)$, where $T$ is trace length and $B$ is block size. Long traces can dominate this budget. This is a generation-token estimate. End-to-end comparisons need to count reference scoring, expert-prompt prefill, and GPU time for training too. [Paper §4.3, Equation 9](https://arxiv.org/html/2610.02140v1#S4.SS3)

## How each baseline uses the expert answer

The expert solution can supply target tokens, condition a rewrite, help a teacher score student prefixes, or make hard rollouts easier to discover. The methods differ in when they generate answers and what gets passed to the loss. [Paper §2 and §5.1](https://arxiv.org/html/2610.02140v1#S5.SS1)

| Method                             | Where the expert signal enters                                  | Where answers come from                               | Training objective                       |
| ---------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------- |
| Expert SFT                         | The complete answer is the target                               | Fixed expert traces                                   | Token cross-entropy                      |
| Rewrite SFT                        | The expert answer conditions one rewrite                        | Fixed reference-model rewrites                        | Token cross-entropy                      |
| Projection sampling + SFT          | The expert answer conditions repeated proposals                 | Offline suffix resampling by a fixed reference model  | Token cross-entropy                      |
| On-policy self-distillation (OPSD) | An expert-conditioned teacher scores student prefixes           | Online, unassisted student answers                    | Token-distribution matching              |
| GRPO                               | A verifier provides rewards in the math setup                   | Online groups of unassisted policy answers            | Group-relative policy updates            |
| UFT                                | Expert prefixes guide exploration and supply supervised targets | Online completion with decreasing amounts of guidance | Hint likelihood plus reward optimization |

OPSD pays for student generation and teacher scoring during training. Its advantage is access to the prefixes the student actually reaches. A teacher that understands the demonstration can provide feedback there, even after an imperfect opening. Shenfeld et al. call their method SDFT and derive reverse-KL distillation, with forward KL performing best in their practical setup. The sampling paper cites both Shenfeld et al. and Zhao et al. and reports using Shenfeld et al.'s hyperparameters. Reproduction requires specifying the teacher update rule and KL direction. [Shenfeld et al., §3](https://arxiv.org/html/2601.19897#S3) · [Zhao et al.](https://arxiv.org/abs/2601.18734) · [Sampling paper §5.1](https://arxiv.org/html/2610.02140v1#S5.SS1)

GRPO estimates advantages from rewards within each group. A group whose answers all get the same reward provides no relative reward signal. UFT uses demonstration prefixes to improve exploration and learns from the prefix tokens through a supervised term. POPE is another relevant guided-exploration comparison. It mixes problems with and without short expert hints and applies RL to the generated answers. DFT and ASFT act at yet another point, adjusting the SFT loss on fixed demonstrations. These last methods offer useful controls for future tests and have no row in the sampling paper's results table. [GRPO](https://arxiv.org/html/2402.03300v3#S4.SS1) · [UFT](https://arxiv.org/html/2505.16984v2#S3) · [POPE](https://arxiv.org/abs/2601.18779) · [DFT](https://arxiv.org/abs/2508.05629) · [ASFT](https://arxiv.org/abs/2509.23753)

## What the experiments establish

The table reports single-answer accuracy for Qwen2.5-3B, in percent. Math mean is the equal-weight average of the four math columns. Prior mean averages the paper's chemistry, MMLU, and GPQA results. [Paper Table 1](https://arxiv.org/html/2610.02140v1#S5.SS2)

| Method              | MATH 3–5 |  AMC | MATH500 | GSM8K | Math mean | Prior mean |
| ------------------- | -------: | ---: | ------: | ----: | --------: | ---------: |
| Base                |     31.5 | 13.3 |    24.5 |  57.9 |      31.8 |       42.2 |
| Expert SFT          |     24.3 | 10.0 |    16.8 |  45.5 |      24.2 |       38.9 |
| OPSD                |     26.7 |  8.4 |    33.2 |  52.4 |      30.2 |       40.4 |
| GRPO                |     45.7 | 24.9 |    31.3 |  80.8 |      45.7 |       41.4 |
| UFT                 |     47.0 | 29.3 |    29.7 |  74.6 |      45.2 |       42.1 |
| Sampling SFT        |     49.5 | 27.7 |    58.2 |  78.2 |      53.4 |       42.0 |
| Sampling SFT → GRPO |     54.5 | 24.1 |    65.2 |  83.0 |      56.7 |       43.0 |

The largest math gain over GRPO is on MATH500, at 26.9 percentage points. GRPO scores higher on GSM8K than standalone sampling SFT, and UFT has the highest AMC score. Starting GRPO from sampling SFT yields the highest math mean, with AMC falling from 27.7% to 24.1%. Two comparisons would sharpen the interpretation. Direct rewrite SFT would help measure the contribution of repeated sampling. Expert SFT followed by the same GRPO recipe would isolate the effect of the initialization data. [Paper §5.2, Table 1](https://arxiv.org/html/2610.02140v1#S5.SS2)

The ordering changes across models and tasks. On the Qwen2.5-7B-Instruct chemistry task, sampling SFT reaches 66.0% target accuracy versus 61.8% for OPSD, with prior-capability averages of 58.6% and 56.8%. For the Olmo chemistry task, OPSD has the higher target score, 59.7% versus 58.3%, and sampling SFT has the higher prior mean, 61.7% versus 60.4%. On open-ended medical question answering graded by GPT-5-mini, OPSD scores 46.6% on the target task to sampling SFT’s 45.8%. For the prior mean, the respective scores are 50.1% and 51.6%. These comparisons make model–data fit worth testing. One Appendix B experiment trains Qwen on traces sampled for Olmo and reports 57.33% chemistry accuracy, versus 66.0% with Qwen-matched traces. That comparison also leaves room for changes in rewrite quality or length. [Table 1](https://arxiv.org/html/2610.02140v1#S5.SS2) · [Appendix B, Table 2 and reference-model comparison](https://arxiv.org/html/2610.02140v1#A2)

The Olmo experiment shows gains that persist as the authors allow more answers per question on their chemistry evaluation. That is stronger evidence than a single-answer score for improvement at the tested inference budgets. For an individual question, zero successes in 64 independent base-model attempts still permits a true per-attempt success probability of about 4.6% at a one-sided 95% confidence limit. The study documents changes in observed capabilities at specific budgets. [Paper Figure 4](https://arxiv.org/html/2610.02140v1#S5.SS3)

## The missing experiments

Start with the math setup. Hold the checkpoint, training problems, expert solutions, loss, and supervised target-token budget fixed. Compare unchanged expert traces, one rewrite, independent best-of-$n$ rewrites, and iterative suffix resampling. Match the generation budget for the two multi-proposal arms and account for rejected candidates, long expert prompts, and failed generations. Run both arms with a shared training-only verifier in a separate comparison. The resulting gap tests whether editing a current candidate yields better training targets than generating several independent rewrites for the same cost. [Paper §4.3 and Appendix C](https://arxiv.org/html/2610.02140v1#A3)

Then run expert SFT and projection-sampled SFT through the same GRPO stage, with the same reward, questions, and rollout budget. Measure both checkpoints before and after GRPO. Report a standalone GRPO arm at comparable end-to-end cost. For both comparisons, freeze evaluation decoding and check question-level overlap. MATH500 was drawn from MATH. If the selected evaluation splits overlap, averaging them gives those questions extra weight. Use paired intervals on the same test questions and report variation across independent training runs. Keep prior-capability scores by benchmark to see where any losses concentrate. [MATH500 split](https://github.com/openai/prm800k#math-splits) · [Paper evaluation setup](https://arxiv.org/html/2610.02140v1#S5.SS1)

A useful result would show where changing the data pays back the cost of generating it. If matched-budget independent rewrites close the gap, a simpler offline pipeline can deliver the benefit. If iterative resampling wins mostly on questions where the base model seldom succeeds, the expert trajectory is likely buying access to valuable solutions. If the gain requires a matched reference model even after controlling rewrite quality, the reference checkpoint and tokenizer version become part of the dataset’s provenance. Changing the learner would then be a reason to revalidate the training set.
