"""Local Hugging Face scoring + frozen weighted-CE integration reference.

No downloads, generation, or optimizer steps. Requires torch + transformers.
Input is the EXACT tokenized training record, including chat template:
{"id":"...", "input_ids":[...], "loss_mask":[0,0,1,...],
 "step_ids":[-1,-1,0,...], "quality":1.0}

Keeping tokenization in the user's existing collator avoids silently changing
chat templates, answer boundaries, EOS handling, or tool-observation masks.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from experiment import frontier, sigmoid


def validate_record(row):
    ids, mask = row['input_ids'], row['loss_mask']
    if len(ids) < 2 or len(ids) != len(mask):
        raise ValueError('input_ids and loss_mask must have the same length >= 2')
    if any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('input_ids must be nonnegative integers')
    if mask[0] != 0 or any(x not in (0, 1) for x in mask) or not any(mask):
        raise ValueError('loss_mask must be binary, start with 0 and contain supervised tokens')
    steps = row.get('step_ids', [0 if m else -1 for m in mask])
    if len(steps) != len(ids) or any(type(s) is not int or (m and s < 0) for m, s in zip(mask, steps)):
        raise ValueError('step_ids must align, with nonnegative IDs at supervised positions')
    quality = row.get('quality', 1.0)
    if not isinstance(quality, (int, float)) or not math.isfinite(quality) or not 0 <= quality <= 1:
        raise ValueError('quality must be a finite number in [0, 1]')
    return steps


def build_weights(scored, method='frontier', level='step', center=-1.3, width=0.6, floor=0.1):
    """Return raw weights aligned with input_ids[1:], zero outside loss_mask.

    Full-batch denominator belongs in the trainer. Per-example normalization
    would cancel sequence weights, so it is deliberately not done here.
    """
    if method not in {'uniform', 'probability', 'monotone', 'frontier'}:
        raise ValueError('unknown method')
    if level not in {'token', 'step', 'sequence'}:
        raise ValueError('unknown level')
    if width <= 0 or not 0 <= floor <= 1:
        raise ValueError('width > 0 and floor in [0,1] required')
    rows = scored['tokens']
    weights = [0.0] * (len(scored['input_ids']) - 1)
    groups = defaultdict(list)
    for token in rows:
        key = token['position'] if level == 'token' else token['step_id'] if level == 'step' else 0
        groups[key].append(token)
    quality = scored.get('quality', 1.0)
    for group in groups.values():
        z = sum(t['cll'] for t in group) / len(group)
        if method == 'uniform':
            weight = 1.0
        elif method == 'probability':
            weight = sum(math.exp(t['logp']) for t in group) / len(group)
        elif method == 'monotone':
            weight = floor + (1-floor)*sigmoid(z/width)
        else:
            weight = frontier(z, center, width, floor)
        for token in group:
            weights[token['position']-1] = quality * weight
    return weights


def frozen_weighted_ce(logits, input_ids, attention_mask, raw_weights):
    """Batched token-normalized loss. All tensors use the SAME padded alignment.

    logits [B,L,V], input_ids/attention_mask [B,L], raw_weights [B,L-1].
    At least 2 examples per optimizer batch if comparing sequence weights.
    For microbatch accumulation, normalize against the entire optimizer batch,
    not each microbatch separately. Multi-device training also needs a global
    denominator; this reference deliberately handles one process only.
    """
    import torch
    import torch.nn.functional as F
    if raw_weights.shape != input_ids[:, 1:].shape:
        raise ValueError('raw_weights must have shape [B,L-1]')
    if not bool(torch.isfinite(raw_weights).all()) or bool((raw_weights < 0).any()):
        raise ValueError('weights must be finite and nonnegative')
    shifted = logits[:, :-1, :].float()
    loss = F.cross_entropy(shifted.reshape(-1, shifted.shape[-1]),
                           input_ids[:, 1:].reshape(-1), reduction='none').reshape(raw_weights.shape)
    weights = raw_weights.detach().to(loss) * attention_mask[:, 1:].to(loss)
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise ValueError('optimizer batch has no positive supervised weight')
    return (weights * loss).sum() / denominator


def score_record(model, row, device, torch, chunk_size=64):
    steps = validate_record(row)
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
        positions = [i for i, m in enumerate(row['loss_mask']) if i > 0 and m]
        tokens = []
        for start in range(0, len(positions), chunk_size):
            pos = positions[start:start+chunk_size]
            index = torch.tensor([i-1 for i in pos], device=device)
            logp = torch.log_softmax(logits.index_select(0, index).float(), dim=-1)
            targets = torch.tensor([row['input_ids'][i] for i in pos], device=device)
            selected = logp.gather(1, targets[:, None]).squeeze(1)
            h = -(logp.exp()*logp).sum(-1)
            for i, lp, ent in zip(pos, selected.cpu().tolist(), h.cpu().tolist()):
                if not math.isfinite(lp+ent):
                    raise ValueError('non-finite score')
                tokens.append(dict(position=i, token_id=row['input_ids'][i], step_id=steps[i],
                                   logp=lp, entropy=ent, cll=lp+ent))
    # Diagnostic only: if the mask omits past assistant actions, or observations
    # are stochastic, this is not the probability of reaching an agent state.
    prefix_logp, prefix_cll = 0.0, 0.0
    for token in tokens:
        token['prior_supervised_logp_sum'] = prefix_logp
        token['prior_supervised_cll_sum'] = prefix_cll
        prefix_logp += token['logp']
        prefix_cll += token['cll']
    result = dict(row, step_ids=steps, tokens=tokens,
                  mean_cll=sum(t['cll'] for t in tokens)/len(tokens),
                  mean_logp=sum(t['logp'] for t in tokens)/len(tokens))
    for name in ['uniform', 'probability', 'monotone', 'frontier']:
        result['weights_'+name] = build_weights(result, method=name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='existing local checkpoint directory')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='cpu, cuda, or mps')
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error('--model must be an existing local directory')
    if args.out.resolve() == args.data.resolve() or args.out.exists():
        parser.error('--out must be a new path distinct from --data')
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit('This adapter requires torch and transformers in the model environment.') from exc
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                trust_remote_code=False).to(args.device).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with args.data.open() as src, args.out.open('x') as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in seen:
                raise ValueError('duplicate example ID')
            seen.add(row['id'])
            if len(row['input_ids']) > args.max_length:
                raise ValueError(f"Example {row['id']} exceeds max length; refusing silent truncation")
            scored = score_record(model, row, args.device, torch)
            scored['scorer'] = dict(checkpoint=str(Path(args.model).resolve()),
                                    torch=torch.__version__, transformers=transformers.__version__,
                                    raw_distribution='temperature=1; no top-p/top-k')
            dst.write(json.dumps(scored, ensure_ascii=False)+'\n')
    print(f'Scored {len(seen)} examples into {args.out}')


if __name__ == '__main__':
    main()
