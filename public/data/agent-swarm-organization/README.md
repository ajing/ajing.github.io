# Agent swarm organization: public companion

Companion to “What Makes an Agent Swarm Work? Simple Rules, Information
Boundaries, and the Value of Verification” by Jing Lu, September 5, 2026.

Article: https://ajing.github.io/posts/2026-09-05-agent-swarm-organization-simple-rules/

## Files

- `results.json`: compact per-trajectory outcomes and recomputed aggregates for
  the 32-trajectory visibility intervention and 40-output final-action study;
  exact utility certificate parameters; study scope and accounting notes.
- `check_theory.py`: Python standard-library checks for small coverage games,
  the visibility cycle, resource-pattern lower-bound certificates, 24 exact
  Bayesian complementary-information examples, and published count arithmetic.
- `plot_results.py`: figure generation from `results.json`; requires Matplotlib.

Download `results.json` and `check_theory.py` into the same directory, then run:

```sh
python3 check_theory.py
```

The checker is a standalone publication companion prepared after collection.
Its 784 unit-weight coverage games are a smaller check than the original
14,400-game study. Its certificate inequalities recheck lower bounds; the
original tight-witness games are not reconstructed here. No calls to a model,
network, or paid service are made.

## Empirical scope

The visibility study used eight question IDs fixed before collection, one
trajectory per policy, with a common generated prefix for P/E/H. P restricted
exploration visibility but pooled raw evidence at final synthesis. The
sequential S baseline used its own adaptive history. Two S intermediate tool
protocol failures were retained.

The final-action study reused these eight development questions after their
results had been observed. B and V each have two repetitions per question;
these are not sixteen independent questions. Both received the same seven-tool
prefix and shared new diagnosis. B could choose verification; V prioritized
it with fallback. N has one repetition and lower actual resource use. One
intermediate protocol failure in each B/V arm was retained.

`outcome` distinguishes correct reference-matching answers, wrong submissions,
and the explicit UNKNOWN response. Exported outcomes preserve the recorded
primary scoring; both scoring procedures agreed. Correct answers matched the
reference under conservative normalization; other candidates were judged
blind to experimental condition. Scoring was not a complete independent
audit of real-world truth or every relationship in the question.

The requested model alias was gpt-5.4-mini with low reasoning effort, without
an immutable provider snapshot or controllable provider seed. Retrieval used
SQLite FTS5 BM25 over the frozen 100,195-document BrowseComp-Plus corpus, with
up to four results and 2,200 characters per extracted passage. This is not
the official retrieval/grader configuration or a leaderboard reproduction.

Original benchmark and corpus instructions:
https://github.com/texttron/BrowseComp-Plus

## Accounting and reproduction boundary

Complete-policy tokens include the cost of reused prefixes for each policy;
physically collected tokens count shared calls once. These measures must not
be added together. Usage includes reported cached input and is not a dollar
estimate. Per-policy admission caps were 192,000 input-plus-output tokens and
eight tool requests; equal caps are not equal realized cost.

The public extract excludes raw source text, source-corpus redistribution,
model session logs, and provider/environment metadata. It supports independent
arithmetic checks of the disclosed outcomes and exact mathematical examples.
It does not enable independent replay or source-level audit of the entire
LLM collection. SHA-256 references identify the local frozen source result
files used to produce the extract; a hash alone is not verification of an
unpublished file's contents.

The full empirical sequence remains exploratory. The later relation controls,
fixed-input repetitions, and final-action study were post hoc development.
They do not certify a universal best topology or sound future semantic
verification. No emergent-role policy training was performed.
