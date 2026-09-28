# Offline SFT selection: reproducible teaching examples

Companion to "What Makes an SFT Example Worth Learning? Quality, Novelty, and Learnability."

## Files

- `experiment.py`: standard-library synthetic binary-policy experiment.
- `coding_examples.py`: four handwritten duplicate-detection implementations and ten declared test cases.
- `test_experiment.py`: numerical-gradient, CLL, masking, and weight checks.
- `llm_adapter.py`: optional local Hugging Face scoring and weighted-loss reference. Its pure Python weight logic is tested; its torch model forward/backward has not been run in this experiment.
- `results.json`: configuration, each seed/method result, bucket diagnostics, and numerical examples.
- `summary.csv`: aggregate results.

Download these files into the same directory. With Python 3:

```bash
python3 coding_examples.py
python3 -m unittest -v test_experiment.py
python3 experiment.py --out reproduced-results
```

No third-party dependency is needed for those three commands. The model adapter
requires torch, transformers, and a local model only when actually scoring an LLM
or calling its PyTorch loss function. No model is downloaded automatically.

## What ran

Five seeds, nine methods, 200 SFT steps and 200 exact expected-reward steps per
run. Each minibatch has 32 examples. Scores are frozen at initialization. The
binary policy has three parameters; it is not a language model. Hyperparameters
are declared in `CONFIG`, not selected using the evaluation results. The
`make_examples` arithmetic uses a sum-of-integers problem; the blog applies the
same deliberately specified three-action probabilities to code strategies.

The reported +/- values are sample standard deviations over seeds, not
confidence intervals. The reward stage analytically enumerates two actions; it
is not sampled PPO/GRPO. The bucket comparisons are not covariate-matched and
have fewer unique examples than full-data methods. No Qwen training result or
advantage of frontier weighting is claimed.

## LLM integration contract

The adapter consumes the exact input IDs, supervision mask, and step IDs from
the original training pipeline. It deliberately does not invent a chat template.
Weights align with `input_ids[1:]`; non-supervised positions receive zero weight.
The loss normalizes over the whole optimization batch. Sequence weighting with
one example per normalized batch cancels out. Gradient accumulation and
distributed use require a consistent global denominator and are not implemented
by this single-process reference.

The adapter's prior-supervised-token likelihood sum is a diagnostic, not a
general estimator of agent-state visitation probabilities.
