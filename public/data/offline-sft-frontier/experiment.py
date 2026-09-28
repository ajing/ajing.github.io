"""Executable teaching experiment; Python standard library only, no LLM claims.

Run: python3 experiment.py --out results
All scores are frozen at initialization. Hyperparameters are illustrative,
predeclared below, and never selected on evaluation results.
"""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


CONFIG = dict(n_train=1000, n_eval=1200, batch_size=32, sft_steps=200,
              reward_steps=200, lr=0.15, seeds=[11, 22, 33, 44, 55],
              initial_weights=[1.6, 0.0, 0.0], target_weights=[0.8, 1.2, -0.7],
              frontier_center=-0.6, frontier_width=0.6, floor=0.1,
              monotone_temperature=0.5)


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)


def cll(probs, action):
    return math.log(probs[action]) + entropy(probs)


def frontier(z, center=-0.6, width=0.6, floor=0.1):
    if width <= 0 or not 0 <= floor <= 1:
        raise ValueError('width > 0 and 0 <= floor <= 1 required')
    return floor + (1 - floor) * math.exp(-0.5 * ((z - center) / width) ** 2)


def correct_prob(weights, row):
    return sigmoid(row['y'] * dot(weights, row['x']))


def make_data(seed, n, ood=False):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        x = [rng.gauss(0, 1) for _ in range(3)]
        if ood:
            x = [0.5 * x[0], 1.5 * x[1], 1.5 * x[2]]
        y = 1 if dot(CONFIG['target_weights'], x) >= 0 else -1
        row = dict(id=i, x=x, y=y, quality=1)
        p = correct_prob(CONFIG['initial_weights'], row)
        row.update(p0=p, z0=cll([p, 1 - p], 0))
        rows.append(row)
    # Names intentionally do not label high CLL as easy or low CLL as novel.
    for rank, row in enumerate(sorted(rows, key=lambda r: (r['z0'], r['id']))):
        row['cll_bucket'] = min(4, rank * 5 // n) + 1
    return rows


def sample_weight(row, method):
    if method == 'uniform' or method.startswith('bucket_'):
        return 1.0
    if method == 'probability':
        return row['p0']
    if method == 'cll_monotone':
        return CONFIG['floor'] + (1 - CONFIG['floor']) * sigmoid(
            row['z0'] / CONFIG['monotone_temperature'])
    if method == 'cll_frontier':
        return frontier(row['z0'], CONFIG['frontier_center'],
                        CONFIG['frontier_width'], CONFIG['floor'])
    raise ValueError(method)


def sft_gradient(weights, batch, method):
    raw = [sample_weight(row, method) for row in batch]
    denom = sum(raw)
    grad = [0.0] * len(weights)
    for row, weight in zip(batch, raw):
        # d[-log(sigmoid(y*w.x))]/dw = -(1-p_correct)*y*x.
        coef = -weight * (1 - correct_prob(weights, row)) * row['y'] / denom
        for j, value in enumerate(row['x']):
            grad[j] += coef * value
    ess = denom ** 2 / sum(weight ** 2 for weight in raw)
    return grad, ess


def reward_gradient(weights, batch):
    # Exact contextual-bandit reward gradient, enumerating both actions.
    # This is an oracle teaching control, NOT sampled PPO/GRPO or an LLM run.
    grad = [0.0] * len(weights)
    for row in batch:
        p = correct_prob(weights, row)
        coef = p * (1 - p) * row['y'] / len(batch)
        for j, value in enumerate(row['x']):
            grad[j] += coef * value
    return grad


def evaluate(weights, data):
    probs = [correct_prob(weights, row) for row in data]
    return dict(accuracy=sum(p >= 0.5 for p in probs) / len(probs),
                expected_reward=statistics.mean(probs),
                mean_entropy=statistics.mean(entropy([p, 1 - p]) for p in probs))


def train_one(train, id_eval, ood_eval, method, seed):
    rng = random.Random(seed + 10000)
    pool = train
    if method.startswith('bucket_'):
        bucket = int(method.split('_')[1])
        pool = [row for row in train if row['cll_bucket'] == bucket]
    weights = CONFIG['initial_weights'][:]
    ess_values = []
    for _ in range(CONFIG['sft_steps']):
        batch = rng.choices(pool, k=CONFIG['batch_size'])
        grad, ess = sft_gradient(weights, batch, method)
        weights = [w - CONFIG['lr'] * g for w, g in zip(weights, grad)]
        ess_values.append(ess)
    result = dict(seed=seed, method=method, unique_pool_size=len(pool),
                  sft_weights=weights[:], mean_batch_ess=statistics.mean(ess_values),
                  sft_id=evaluate(weights, id_eval), sft_ood=evaluate(weights, ood_eval))
    # All methods see exactly the same subsequent prompt minibatches per seed.
    rng = random.Random(seed + 20000)
    for _ in range(CONFIG['reward_steps']):
        batch = rng.choices(train, k=CONFIG['batch_size'])
        grad = reward_gradient(weights, batch)
        weights = [w + CONFIG['lr'] * g for w, g in zip(weights, grad)]
    result.update(reward_weights=weights, reward_id=evaluate(weights, id_eval),
                  reward_ood=evaluate(weights, ood_eval))
    return result


def make_examples():
    p = [0.8, 0.15, 0.05]
    labels = ['逐项累加 1 到 100', '首尾配对，50 对，每对 101', '用数学归纳法证明求和公式']
    actions = []
    for i, label in enumerate(labels):
        z = cll(p, i)
        w = frontier(z, center=-1.3, width=0.6, floor=0.1)
        actions.append(dict(action=label, p=p[i], entropy=entropy(p), cll=z,
                            frontier_raw_weight=w, ce_chosen_logit_magnitude=1-p[i],
                            weighted_chosen_logit_magnitude=w*(1-p[i])))
    # Both trajectories have identical final local conditionals.
    paths = []
    for label, prefix in [('常见前缀', [0.8, 0.8]), ('罕见前缀', [0.02, 0.8])]:
        local = [0.9, 0.1]
        paths.append(dict(name=label, prefix_conditionals=prefix,
                          prefix_probability=math.prod(prefix),
                          final_action_probability=local[0], final_action_cll=cll(local, 0),
                          full_trajectory_probability=math.prod(prefix)*local[0]))
    q = [1.0, 0.0, 0.0]
    kl = sum(qi * math.log(qi / pi) for qi, pi in zip(q, p) if qi > 0)
    return dict(provenance='HAND-SPECIFIED SEMANTIC ACTION PROBABILITIES, NOT LLM SCORES',
                question='计算 1+2+…+100，三个方法均可得到 5050。',
                actions=actions, paths=paths,
                off_policy_counterexample=dict(p=p, q=q, expected_cll=cll(p, 0),
                                               entropy_identity=entropy(p)-entropy(q)-kl),
                uniform_counterexample=dict(p=[1/3]*3, cll=[cll([1/3]*3, i) for i in range(3)]))


def summarize(runs):
    rows = []
    for method in dict.fromkeys(run['method'] for run in runs):
        selected = [r for r in runs if r['method'] == method]
        row = dict(method=method)
        for stage in ['sft_id', 'sft_ood', 'reward_ood']:
            for metric in ['accuracy', 'expected_reward']:
                values = [r[stage][metric] for r in selected]
                name = f'{stage}_{metric}'
                row[name + '_mean'] = statistics.mean(values)
                row[name + '_sd'] = statistics.stdev(values)
        row['mean_batch_ess'] = statistics.mean(r['mean_batch_ess'] for r in selected)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    methods = ['uniform', 'probability', 'cll_monotone', 'cll_frontier'] + [f'bucket_{i}' for i in range(1, 6)]
    runs, diagnostics, initial = [], [], []
    for seed in CONFIG['seeds']:
        train = make_data(seed, CONFIG['n_train'])
        id_eval = make_data(seed + 100, CONFIG['n_eval'])
        ood_eval = make_data(seed + 200, CONFIG['n_eval'], ood=True)
        initial.append(dict(seed=seed, id=evaluate(CONFIG['initial_weights'], id_eval),
                            ood=evaluate(CONFIG['initial_weights'], ood_eval)))
        for bucket in range(1, 6):
            group = [r for r in train if r['cll_bucket'] == bucket]
            diagnostics.append(dict(seed=seed, bucket=bucket, n=len(group),
                                    p_mean=statistics.mean(r['p0'] for r in group),
                                    cll_min=min(r['z0'] for r in group),
                                    cll_max=max(r['z0'] for r in group),
                                    x_abs_mean=[statistics.mean(abs(r['x'][j]) for r in group) for j in range(3)]))
        for method in methods:
            runs.append(train_one(train, id_eval, ood_eval, method, seed))
    summary = summarize(runs)
    payload = dict(config=CONFIG, examples=make_examples(), initial=initial,
                   bucket_diagnostics=diagnostics, runs=runs, summary=summary,
                   scope='Synthetic binary contextual bandit. No LLM, no actual GRPO/PPO. '
                         'Bucket comparisons are exploratory and not covariate-matched.')
    (args.out / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    with (args.out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ['# 实际运行结果：合成二元策略模型', '',
             '数值是 5 个种子的均值 ± 样本标准差，单位为 %；不是置信区间。', '',
             '| 方法 | SFT ID accuracy | SFT OOD accuracy | SFT OOD expected reward | 奖励优化后 OOD expected reward |',
             '|---|---:|---:|---:|---:|']
    for row in summary:
        cells = [row['method']]
        for key in ['sft_id_accuracy', 'sft_ood_accuracy', 'sft_ood_expected_reward', 'reward_ood_expected_reward']:
            cells.append(f"{100*row[key+'_mean']:.2f} ± {100*row[key+'_sd']:.2f}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '所有方法使用相同初始参数；每阶段 200 次更新，每批 32 条。',
              '奖励优化使用二元动作的精确期望梯度，不代表真实 LLM 的 RL 训练结果。',
              '分桶组只用 200 条不同样本，完整数据组用 1000 条；曝光次数相同，但独特数据量不同。',
              '桶之间特征分布未匹配，不能据此断言 CLL 对学习价值有因果影响。']
    (args.out / 'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
