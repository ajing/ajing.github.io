---
author: Jing Lu
pubDatetime: 2026-11-23T22:45:00-08:00
title: "Why All Data Vendors Are Bad"
featured: false
draft: false
tags:
  - AI
  - LLM Agents
  - Post Training
  - Evaluation
description: "A data vendor can pass every acceptance test and still deliver a batch the model has outgrown. Here is what I would buy instead."
---

What was Microsoft’s Intelligent Cloud operating margin in fiscal 2024, using the segment definitions it adopted in fiscal 2025?

The 2024 filing reports 49.584 billion dollars in operating income on 105.362 billion dollars in revenue. Divide one by the other and you get 47.1%. In its 2025 filing, Microsoft recalculated the 2024 figures using its new segment definitions. That gives you 43.2%, from 37.813 billion in operating income and 87.464 billion in revenue. The fiscal year and segment name are the same. Which businesses count toward the segment depends on the filing. [Microsoft’s FY2024 segment results](https://www.microsoft.com/en-us/investor/earnings/fy-2024-q4/segment-revenues) · [FY2025 segment results](https://www.microsoft.com/en-us/investor/earnings/fy-2025-q4/segment-revenues) · [FY2025 annual report, Note 1](https://www.microsoft.com/investor/reports/ar25/)

Now imagine you run a lab training a financial research agent. The agent gives you 47.1%. You ask a vendor for 2,000 cases like this one, each hard enough to make the model fail. By the time the analysts deliver, a separate training run has taught your agent to look for the figures in the later filing. The vendor has delivered accurate work on time. If you were placing the order today, you’d spend the money on a different problem.

That’s why I’m skeptical of buying large, fixed batches of data to improve an existing model. A lab can put the number of accepted tasks on a purchase order. It can only find out what those tasks teach a model by measuring what happens after training.

## Which month’s SOTA did we agree on?

Suppose the lab gives the vendor a checkpoint, a saved version of the model at a particular stage of training. It also shares the tools the model uses and the grader that scores its answers. The vendor can test the brief on day one. Which model will judge the work eight weeks later?

If you grade against the signing-day model, the vendor can earn the full fee for a batch the lab would no longer order. If you grade against the delivery-day model, the vendor can lose the full fee for work that passed every test available when they started. “Make it hard for the state of the art” leaves both sides guessing until you name a model version and agree on how it will run.

The client can specify good financial analysis today. Use the same segment definition for every figure in a calculation. Cite the source figures. If the filings lack the data to calculate a margin, explain what is missing. Those requirements can anchor the contract. The lab has to wait for the next model before it can measure which of those rules the model still breaks.

Microsoft Research built that measurement into the training of FrogNano, a coding agent. Its team ran candidate tasks against the current checkpoint and used the attempts to select or revise the tasks. When scores on its validation set leveled off, it used the best checkpoint to generate and calibrate the next round. [FrogNano, §3](https://arxiv.org/html/2609.07925v5#S3)

The economics change when people have to collect the cases. A researcher can change priorities in a meeting. The vendor may need to recruit different specialists and teach reviewers a different rubric. Someone also has to pay for the cases already under way.

## Which model did you stump?

A vendor can run a strong public model and still get the wrong brief. Suppose the public model uses the old Microsoft filing. Your lab’s checkpoint finds the right table but divides segment operating income by company-wide revenue. Both models fail. The first needs practice choosing the right filing. The second needs practice matching the two figures in the calculation. If you order from the public model’s failure log, you can spend two months teaching a skill your agent already knows.

Model access also affects the cost of making synthetic data, which models generate. A single attempt can show a failure. Eight let you see whether it repeats and look for successful solutions the grader missed. At eight attempts each, a 50-task pilot takes 400 runs just to measure how often the model succeeds. It takes more calls to generate candidate tasks, write worked solutions, and check them. A lab with its own checkpoints and the capacity to run them can budget for those calls directly. If a vendor gets a fixed price per accepted task, every discarded run comes out of that price.

For Nemotron 3.5, NVIDIA generated analyst-style questions from SEC filings and took multiple samples to check that they were answerable. Teacher models then solved validated questions using the same tools available during training. NVIDIA used measured difficulty to choose between training on worked examples and training with rewards. The report also names the intermediate checkpoint that NVIDIA used to measure difficulty for a separate set of agent tasks. Access to these models helped the team decide what data to use and how to use it. [Nemotron 3.5 technical report, §§3.2–3.3.1](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf#page=16)

## What did the failure tell you?

Say the model gets the Microsoft question right. The analyst can spend the afternoon finding another filing where the reporting basis matters. Or they can add a requirement to cite eight documents when two settle the question. Under a contract that pays for accepted failures, either move might earn the same amount. Only one gives the lab another case worth practicing.

The grader can create the same problem without anyone trying to make the task harder. In their 2026 revision of SWE-bench Pro, Scale and Reflection corrected 69 tasks whose instructions contradicted their tests. A model can fail that kind of task by following the prompt. A score of zero tells you where to look. You still have to read the attempt and check what the test actually required. [SWE-bench Pro V2 release](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2)

Even a well-graded task can be a poor fit for the planned training run. Some reinforcement learning methods compare attempts at the same task to decide which ones to reinforce. If the only reward is pass or fail, eight failures earn the same score. The update gets no signal from the difference in rewards because there is no difference. If three attempts succeed, the training method can favor those attempts. DAPO filters out groups that are all correct or all incorrect. A task the model fails every time can still serve as an evaluation or come with a worked example for training. [DAPO, §3.2](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)

## Who gets paid to cancel the next 1,950 tasks?

I’d buy the first 50 and make the rest optional.

1. Agree on the work and the test. Ask for source passages, worked solutions, and grading rules that accept defensible answers. If difficulty affects payment, agree on the checkpoint, tools, grader, and number of attempts before work starts. Pay against those terms.
2. Retest before placing the next order. Run a sample on the latest checkpoint with the same tools, grader, and number of attempts. The lab can run a private model itself and share records of the attempts. Pay the analysts to diagnose the failures. If the agent invents a margin when segment revenue is missing, buy cases that test when to withhold a number. Include answerable cases to catch an agent that declines to answer too often.
3. Measure what the model learned. Test on companies left out of training. To estimate what the pilot contributed, compare training runs that start from the same checkpoint and use the same budget. Use the pilot data in one run and the lab’s usual training data in its place in the other. Share enough of the results with the vendor to decide whether to order more.

The vendor’s best contribution may be the analyst who brings in an original workbook, the reviewer’s note explaining why a plausible answer was rejected, and a fresh case that tests whether the model learned the right lesson. I’d pay that analyst to tell me when we have enough examples of a mistake.

My first question for a data vendor would be, “What would convince you to stop making these?” My first question for the lab would be, “Would you pay them for finding out?”
