# Blog numerical checks — September 6, 2026

Run `python3 numerical_checks.py` (Python 3, standard library only). The script
writes `results.json` and `lora-factor-error.svg` into its own directory.

These are synthetic algebra and estimator checks. No LLM is trained or queried,
and no downstream model accuracy is measured.

- **LoRA factor coordinates:** 100 seeds, six fixed reparameterizations each,
  16 × 16 effective updates of rank 4. Both adapters implement the same update.
  Compare averaging factors with averaging materialized updates. Verify the
  measured relative Frobenius error against its analytic value. The figure
  shows the median over seeds; this is an algebra check, not a statistical
  estimate of expected model accuracy. Scaling values were chosen to illustrate
  the identity and a sign-flip counterexample, not sampled from real adapters.
- **GAE:** 400 cases (100 seeds × four sequence lengths), compare the backward
  recursion with the explicit discounted sum of TD residuals. True terminal
  value is zero; gamma = 0.99 and lambda = 0.95.
- **Reward projection:** 400 cases, check that adding an equal share of the
  residual to local scores exactly recovers the sequence score within numerical
  tolerance. Conservation alone does not establish faithful causal credit.

Floating-point checks use a tolerance of 1e-12. The SVG is generated from the
same numbers as the JSON. Prettier may reflow the generated JSON/SVG without
changing their values.
