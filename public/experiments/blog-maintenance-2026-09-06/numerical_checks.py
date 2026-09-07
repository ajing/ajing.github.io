"""Reproducible algebra checks, Python 3 standard library only. No model calls.
Run: python3 numerical_checks.py
Outputs: results.json and lora-factor-error.svg next to this script.
"""
from pathlib import Path
import json
import math
import random
import statistics

OUT = Path(__file__).resolve().parent

def mm(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def scale(a, c):
    return [[c*x for x in row] for row in a]

def mean(a, b):
    return [[(x+y)/2 for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

def error(a, target):
    return math.sqrt(sum((x-y)**2 for ra, rb in zip(a, target) for x, y in zip(ra, rb)) / sum(x*x for row in target for x in row))

rows = []
for c in [1, 2, 4, 8, 16, -1]:
    naive, dense = [], []
    for seed in range(100):
        rng = random.Random(seed)
        b = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(16)]
        a = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(4)]
        target = mm(b, a)
        b2, a2 = scale(b, c), scale(a, 1/c)
        dense.append(error(mean(target, mm(b2, a2)), target))
        naive.append(error(mm(mean(b, b2), mean(a, a2)), target))
    expected = abs((2+c+1/c)/4-1)
    assert max(abs(v-expected) for v in naive) < 1e-12
    assert max(dense) < 1e-12
    rows.append(dict(scale=c, seeds=100, factor_relative_error_median=statistics.median(naive), delta_relative_error_max=max(dense), analytic_relative_error=expected))

# Independent forward-sum vs backward-recursion GAE comparison.
max_gae_error = 0.0
max_projection_error = 0.0
for seed in range(100):
    rng = random.Random(seed)
    for length in [1, 2, 8, 32]:
        reward = [0.0]*(length-1) + [rng.uniform(-2, 2)]
        values = [rng.uniform(-1, 1) for _ in range(length)] + [0.0]
        gamma, lam = 0.99, 0.95
        deltas = [reward[t]+gamma*values[t+1]-values[t] for t in range(length)]
        direct = [sum((gamma*lam)**k * deltas[t+k] for k in range(length-t)) for t in range(length)]
        backward = [0.0]*length
        carry = 0.0
        for t in reversed(range(length)):
            carry = deltas[t]+gamma*lam*carry
            backward[t] = carry
        max_gae_error = max(max_gae_error, max(abs(a-b) for a, b in zip(direct, backward)))
        phi = [rng.uniform(-1, 1) for _ in range(length)]
        z = rng.uniform(-2, 2)
        projected = [v+(z-sum(phi))/length for v in phi]
        max_projection_error = max(max_projection_error, abs(sum(projected)-z))
assert max_gae_error < 1e-12
assert max_projection_error < 1e-12
result = dict(scope='Synthetic linear algebra and estimator identity checks; no trained LLM or task benchmark', seeds=list(range(100)), matrix_shape=[16,16], lora_rank=4, precision='Python float (64-bit)', lora=rows, gae_cases=400, gae_max_absolute_error=max_gae_error, projection_cases=400, projection_max_absolute_error=max_projection_error)
(OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')

# A deterministic SVG; vertical error scale starts at zero.
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 470" role="img" aria-labelledby="title desc">', '<title id="title">Equivalent LoRA updates can break factor averaging</title>', '<desc id="desc">Synthetic 16 by 16 matrices, rank 4, 100 seeds. Median relative Frobenius error: scaling 1: 0 percent, 2: 12.5 percent, 4: 56.25 percent, 8: 153.125 percent, 16: 351.5625 percent, sign flip: 100 percent. Averaging materialized deltas has zero error.</desc>', '<rect width="840" height="470" fill="#ffffff"/>','<g font-family="system-ui,sans-serif" fill="#18283b">','<text x="40" y="35" font-size="22" font-weight="700">Same update. Different factors. A broken average.</text>','<text x="40" y="62" font-size="14">Synthetic identity check · 100 seeds · 16 × 16 update, rank 4</text>','<text x="40" y="91" font-size="13">Relative Frobenius error (%) — lower is better</text>']
for tick in [0,100,200,300,400]:
    y=360-tick*0.6
    svg.extend([f'<line x1="80" y1="{y}" x2="810" y2="{y}" stroke="#dce3ea"/>',f'<text x="68" y="{y+5}" text-anchor="end" font-size="13">{tick}</text>'])
for i,row in enumerate(rows):
    x=105+i*118; percent=row['factor_relative_error_median']*100; h=percent*.6
    svg.extend([f'<rect x="{x}" y="{360-h}" width="66" height="{max(h,1)}" fill="#b74736" rx="3"/>',f'<text x="{x+33}" y="{350-h}" text-anchor="middle" font-size="14">{percent:.2f}%</text>',f'<circle cx="{x+33}" cy="360" r="4" fill="#087e8b"/>',f'<text x="{x+33}" y="385" text-anchor="middle" font-size="13">{"sign flip" if row["scale"] == -1 else "c = "+str(row["scale"])}</text>'])
svg.extend(['<text x="80" y="420" font-size="14" fill="#b74736">■ Average A and B separately</text>','<text x="410" y="420" font-size="14" fill="#087e8b">● Average materialized BA updates: 0%</text>','<text x="80" y="449" font-size="12">Both source adapters implement the same BA. This is not an LLM accuracy experiment.</text>','</g></svg>'])
(OUT/'lora-factor-error.svg').write_text('\n'.join(svg)+'\n')
print(json.dumps(result, indent=2))
