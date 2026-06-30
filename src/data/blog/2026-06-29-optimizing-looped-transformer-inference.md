---
author: Jing Lu
pubDatetime: 2026-06-29T18:00:00Z
modDatetime: 2026-06-30T00:45:00Z
title: "Optimizing Inference for Router Looped Transformers"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Inference
  - Systems
  - ML Engineering
description: "A research note on serving router looped transformers: why normal KV cache semantics break, what latency data says so far, and how vLLM or SGLang could be adapted with route-template batching and virtual-step KV cache."
---

Looped transformers change the inference problem in a subtle way.

In a normal decoder-only transformer, layer 17 is always layer 17. During generation, its KV cache can be indexed by request, layer, and token position. A serving system can safely say:

```text
cache[request_id][layer_id][token_position]
```

In a looped transformer, the same physical block can be reused many times. In a router looped transformer, the route can also depend on the input or context. The physical block id is no longer enough.

For example:

```text
route step 0 -> physical block 0
route step 1 -> physical block 1
route step 2 -> physical block 0
route step 3 -> physical block 1
```

The two visits to physical block 0 share weights, but they do not share hidden states. If they share the same KV slot, the cache is wrong.

The correct serving abstraction is closer to:

```text
cache[request_id][route_step][physical_block_id][token_position]
```

or, in serving-engine terms:

```text
virtual_layer_id = route_step * num_physical_blocks + physical_block_id
```

That is the core idea of this note: a looped transformer has fewer unique parameters, but the serving system still needs a virtual layer identity for cache correctness.

## What We Are Testing

The current research model is intentionally small. It uses two reusable transformer blocks and runs them recurrently:

```text
n_unique_layers = 2
n_loops = 4
effective fixed depth = 8 block calls
router_max_steps = 8
d_model = 50
n_heads = 2
```

The router observes a sequence summary:

```text
mean hidden state + last-token hidden state + route-step signal
```

At each route step it chooses among:

```text
physical block 0, physical block 1, exit
```

During training, the router is soft. It evaluates all candidate blocks and mixes outputs by probability. That is useful for gradients, but bad for deployment: soft routing does not actually skip the expensive block executions.

So the inference hypothesis is:

```text
train with soft routing
convert to a hard route at inference
batch requests by route template
store KV cache by virtual route step, not just physical block
```

## The First Result: Soft Routing Is Not an Inference Optimization

On a 300-step single-seed Modal probe, the soft router was slightly more accurate than the fixed loop, but it was slower because it executed roughly twice as many core block calls.

| Model path                    | Eval accuracy | Latency per batch |          Throughput | Core block calls |
| ----------------------------- | ------------: | ----------------: | ------------------: | ---------------: |
| fixed `2x4`                   |    `0.162109` |       `27.568 ms` | `2321.5 examples/s` |          `8.000` |
| soft router `2x4`             |    `0.166016` |       `45.420 ms` | `1409.1 examples/s` |         `16.000` |
| hard router, threshold `0.50` |    `0.164062` |       `34.589 ms` | `1850.3 examples/s` |          `7.982` |
| hard router, threshold `0.30` |    `0.134766` |       `34.253 ms` | `1868.4 examples/s` |          `7.646` |

Source artifact:

```text
runs/modal-downloads/modal_inference_fixed_router_300s_seed0_20260629/inference_report.md
```

The important result is not that the router is already faster. It is not.

The important result is that hard routing kept nearly the same accuracy while reducing executed block calls back toward the fixed-loop budget. This suggests the router should be treated as a training-time search policy and converted into a simpler serving-time plan.

## Multi-Seed Modal Gate: Hard Router Can Beat Fixed Accuracy

The next run used 600 steps and three seeds on the easier transfer setting:

```text
num_nodes = 16
train_max_hops = 3
eval_max_hops = 4
seeds = 0, 1, 2
```

| Inference path       | Mean eval acc |    Acc std | Latency per batch |          Throughput | Mean block calls |
| -------------------- | ------------: | ---------: | ----------------: | ------------------: | ---------------: |
| fixed `2x4`          |    `0.166667` | `0.015666` |       `17.433 ms` | `3673.7 examples/s` |          `8.000` |
| soft router          |    `0.147786` | `0.022637` |       `28.148 ms` | `2274.2 examples/s` |         `16.000` |
| hard router `t=0.45` |    `0.147786` | `0.007394` |       `23.985 ms` | `2674.6 examples/s` |          `7.930` |
| hard router `t=0.50` |    `0.172526` | `0.004066` |       `23.276 ms` | `2750.2 examples/s` |          `7.965` |
| hard router `t=0.55` |    `0.173828` | `0.011880` |       `23.381 ms` | `2737.3 examples/s` |          `7.973` |

Source artifact:

```text
runs/modal-downloads/modal_inference_fixed_router_thresholds_600s_seeds012_20260629/inference_summary.json
```

This is the first useful candidate signal:

```text
hard router t=0.55:
  accuracy beats fixed by about 4.3 percent relative
  block calls stay near 8
  wall-clock latency is still worse than fixed
```

So the architecture signal and systems signal disagree. The router can find a slightly better path, but the current implementation has too much control-flow overhead.

That is exactly where serving optimization matters.

## Harder Transfer Gate

The harder transfer setting used more nodes and a longer evaluation horizon:

```text
num_nodes = 24
train_max_hops = 4
eval_max_hops = 5
seeds = 0, 1, 2
```

| Inference path       | Mean eval acc |    Acc std | Latency per batch |          Throughput | Mean block calls |
| -------------------- | ------------: | ---------: | ----------------: | ------------------: | ---------------: |
| fixed `2x4`          |    `0.104167` | `0.013005` |       `17.966 ms` | `3563.0 examples/s` |          `8.000` |
| soft router          |    `0.100911` | `0.022553` |       `29.026 ms` | `2206.4 examples/s` |         `16.000` |
| hard router `t=0.50` |    `0.091797` | `0.031974` |       `24.325 ms` | `2631.5 examples/s` |          `7.928` |
| hard router `t=0.55` |    `0.101562` | `0.001953` |       `24.506 ms` | `2613.1 examples/s` |          `7.955` |
| hard router `t=0.60` |    `0.108073` | `0.002983` |       `23.699 ms` | `2700.7 examples/s` |          `7.971` |

Source artifact:

```text
runs/modal-downloads/modal_inference_harder_transfer_thresholds_600s_seeds012_20260629/inference_summary.json
```

The current harder-transfer candidate is `t=0.60`. It beats fixed accuracy slightly and keeps fixed-like block calls. It is still slower in wall-clock time, which means the next bottleneck is not model compute. It is routing overhead, dynamic grouping, and cache layout.

## Local Serving Benchmark: Route-Template Replay

The first local benchmark does not implement full KV cache. It measures a simpler serving optimization: freeze the hard route plan and replay it as virtual recurrent layers.

Command:

```bash
uv run python scripts/benchmark_inference_cache.py \
  --repeats 30 \
  --warmup 8 \
  --batch-size 16 \
  --prompt-len 128 \
  --decode-tokens 32 \
  --d-model 64 \
  --n-heads 4 \
  --route-steps 8 \
  --output runs/local-cache-benchmarks/inference_cache_benchmark_template_20260629.json
```

Result:

| Path                  | Mean latency |     Std | Speedup vs generic hard router | Max logit diff |
| --------------------- | -----------: | ------: | -----------------------------: | -------------: |
| generic hard router   |   `8.996 ms` | `0.138` |                        `1.00x` |            n/a |
| route-plan replay     |   `7.527 ms` | `0.114` |                        `1.20x` |          `0.0` |
| route-template replay |   `6.379 ms` | `0.089` |                        `1.41x` |          `0.0` |

Route template in this run:

```text
0,1,0,1,0,1,0,1
```

Virtual cache slots:

```text
(route_step=0, block_id=0)
(route_step=1, block_id=1)
(route_step=2, block_id=0)
(route_step=3, block_id=1)
(route_step=4, block_id=0)
(route_step=5, block_id=1)
(route_step=6, block_id=0)
(route_step=7, block_id=1)
```

The exact-logit match matters. It means route-template replay is not an approximation. It is the same hard-router computation with less runtime overhead.

This suggests a practical serving path:

```text
1. run router once during prefill, or during an early planning stage
2. assign each request a route_template_id
3. group active requests by route_template_id
4. run the corresponding fixed virtual block sequence
```

## Local KV Benchmark: Virtual-Step Cache

The second local benchmark is a synthetic causal decoder microbench. It compares:

```text
no cache:
  for every generated token, rerun virtual blocks over the whole prefix

virtual-step KV cache:
  prefill creates KV for each (route_step, physical_block_id)
  decode computes only the new token and appends to each virtual-step slot
```

Single setting:

| Batch | Prompt len | Decode tokens | Route steps |     No cache | Virtual-step KV | Speedup |
| ----: | ---------: | ------------: | ----------: | -----------: | --------------: | ------: |
|  `16` |      `128` |          `32` |         `8` | `195.321 ms` |    `116.726 ms` | `1.67x` |

Prompt-length matrix:

| Prompt len | Route steps |     No cache | Virtual-step KV | Speedup |
| ---------: | ----------: | -----------: | --------------: | ------: |
|       `64` |         `4` |  `41.401 ms` |     `32.638 ms` | `1.27x` |
|       `64` |         `8` |  `80.746 ms` |     `64.338 ms` | `1.26x` |
|      `128` |         `4` |  `49.207 ms` |     `35.050 ms` | `1.40x` |
|      `128` |         `8` |  `99.023 ms` |     `70.293 ms` | `1.41x` |
|      `256` |         `4` |  `67.680 ms` |     `40.351 ms` | `1.68x` |
|      `256` |         `8` | `132.918 ms` |     `78.141 ms` | `1.70x` |

Source artifacts:

```text
runs/local-cache-benchmarks/inference_cache_benchmark_template_20260629.json
runs/local-cache-benchmarks/decoder_kv_matrix_20260629.json
```

This is CPU synthetic data, not deployment latency. But it supports two design decisions:

- cache payoff grows with prompt length;
- route depth does not remove the value of KV cache;
- the cache key must include route step, otherwise reused physical blocks collide.

## Why vLLM Needs Route Identity

vLLM's PagedAttention design stores KV cache in fixed-size blocks rather than one large contiguous tensor ([vLLM PagedAttention](https://docs.vllm.ai/en/latest/design/paged_attention/)). Its prefix caching design hashes full KV blocks using parent hash, block tokens, and extra hashes such as LoRA ids, multimodal hashes, or cache salts ([vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)).

That "extra hashes" concept is exactly where route identity belongs.

For a looped transformer, the cache hash should include:

```text
model_id
request prefix hash
route_template_id
virtual_layer_id
physical_block_id
route_step
token block ids
```

The minimal vLLM adaptation would be:

```text
1. add looped-transformer metadata to model config
2. expose a route planner that returns route_template_id and virtual_layer_ids
3. include route_template_id / virtual_layer_id in prefix-cache extra hashes
4. map virtual_layer_id to the same physical weights during model execution
5. batch scheduler groups requests by next virtual_layer_id, not only by decode step
```

The likely code areas to study first are:

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/kv_cache_utils.py
```

The key rule is simple:

```text
same tokens + same physical block is not enough
same tokens + same virtual route step is required
```

## Why SGLang Needs Route-Aware Radix Keys

SGLang's RadixAttention keeps KV cache for prompts and generation results in a radix tree, so later requests can reuse matching prefixes. The public SGLang writeup describes this as a mapping from token sequences to KV tensors, with LRU eviction and cache-aware scheduling ([SGLang RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)).

For normal transformers, a token-prefix key is enough because layer order is fixed.

For router looped transformers, the radix key should become route-aware:

```text
radix_key = [
  route_template_id,
  virtual_layer_id,
  token_0,
  token_1,
  ...
]
```

or, if token-level routing is enabled:

```text
radix_key = [
  route_signature_for_token_span,
  virtual_layer_id,
  token_span
]
```

The likely SGLang code areas to study first are:

```text
python/sglang/srt/mem_cache/radix_cache.py
python/sglang/srt/mem_cache/memory_pool.py
python/sglang/srt/managers/schedule_batch.py
```

The serving scheduler should prefer batches with the same next virtual route step. Otherwise, each request may need a different physical block at the same decode position, which destroys batching efficiency.

## Step-by-Step Optimization Plan

The current data suggests this order:

### Step 1: Do not serve the soft router

Soft routing is useful for training, but it doubles block calls in the current implementation:

```text
fixed loop: 8 block calls
soft router: 16 block calls
hard router: about 8 block calls
```

The serving path should use hard routing, top-k routing, or a distilled route policy.

### Step 2: Freeze a request-level route template

The route-template replay benchmark shows `1.41x` speedup over generic hard routing with identical logits. This is the cheapest systems win.

Request-level routing is also easier to implement than token-level routing because every token in the sequence follows the same virtual block plan.

### Step 3: Add virtual-layer KV cache

Use:

```text
virtual_layer_id = route_step * num_physical_blocks + physical_block_id
```

This preserves cache correctness while still reusing the same physical weights.

### Step 4: Batch by next virtual layer

The scheduler should group requests by:

```text
next_virtual_layer_id
route_template_id
decode phase
```

This turns a dynamic router model back into a small set of dense batched kernels.

### Step 5: Only then try token-level routing

Token-level routing is more powerful, but it complicates the cache. Different tokens in the same request can take different routes, so the runtime needs per-token route signatures and may need to compact tokens by route step.

I would not start there. The current data says request-level route-template batching already has measurable upside.

## What Is Still Missing

The current evidence is enough to justify the optimization direction. It is not enough to claim that router looped transformers are already faster in a real serving stack.

Three things are still missing:

| Missing evidence                              | Why it matters                                                                        | Current proxy                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| GPU decode latency with real KV cache         | CPU microbenchmarks can overstate or understate the serving win                       | Synthetic CPU virtual-step KV cache gives `1.27x` to `1.70x` |
| Route-template diversity under varied prompts | Template batching only works if many requests share a small number of route templates | Current local run has one template: `0,1,0,1,0,1,0,1`        |
| Accuracy/latency Pareto across thresholds     | A router candidate should not be chosen by accuracy alone                             | Modal thresholds show `t=0.55` and `t=0.60` are promising    |

The most important missing measurement is:

```text
fixed loop vs hard router vs hard router + template batching + virtual-step KV cache
```

measured on the same GPU, same batch sizes, same prompt lengths, same decode lengths, and same model checkpoint.

## Experiments To Run Next

I would run the next experiments in this order.

| Priority | Experiment                  | What to measure                                              | Pass condition                                                                                   |
| -------: | --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
|     `P0` | GPU decode benchmark        | TTFT, TPOT, end-to-end latency, tokens/s                     | hard-router optimized path is faster than generic hard router and approaches or beats fixed loop |
|     `P0` | Route-template histogram    | number of unique templates, top-k coverage, template entropy | top `8` to `16` templates cover most requests                                                    |
|     `P1` | Threshold Pareto sweep      | accuracy, latency, block calls for `t=0.45` to `0.70`        | one threshold beats fixed accuracy without exceeding fixed block-call budget                     |
|     `P1` | Prompt/decode length matrix | prompt lengths `64/128/256/512`, decode lengths `16/32/64`   | cache speedup grows with prompt length and remains positive at longer decode                     |
|     `P2` | Token-level router scout    | per-token route diversity, batching fragmentation, accuracy  | token routing improves accuracy enough to justify scheduler complexity                           |
|     `P2` | Template distillation       | small route predictor vs original router                     | same template choices with lower planning overhead                                               |

The `P0` experiments are the ones needed before making a strong inference claim. The `P1` experiments choose a usable router candidate. The `P2` experiments are research extensions.

## Low-Cost Modal Plan

Because the goal is to control spend, the next Modal run should be a bounded inference-only benchmark, not a long training run.

I would use:

```text
one small GPU
one checkpoint
three seeds only if the first seed is promising
short prompt/decode grid first
hard timeout
write JSON after every benchmark cell
```

Suggested first grid:

| Setting         | Values                                                                 |
| --------------- | ---------------------------------------------------------------------- |
| batch size      | `1`, `8`, `16`                                                         |
| prompt length   | `64`, `128`, `256`                                                     |
| decode tokens   | `16`, `32`                                                             |
| route steps     | `8`                                                                    |
| inference paths | fixed, soft router, hard router, hard + template, hard + template + KV |

The first stopping rule should be simple:

```text
stop if hard + template is not at least 1.20x faster than generic hard router
stop if virtual-step KV does not beat no-cache at prompt_len >= 128
stop if accuracy drops below fixed by more than one seed-level std
```

That keeps the experiment cheap and prevents a long run from chasing a weak systems signal.

## What Would Make This Publishable As A Research Result

For a stronger public claim, I would want one table like this:

| Model                    |           Accuracy |          TTFT |          TPOT |      tokens/s |            KV memory | Block calls | Notes                    |
| ------------------------ | -----------------: | ------------: | ------------: | ------------: | -------------------: | ----------: | ------------------------ |
| fixed loop               |           baseline |      baseline |      baseline |      baseline |             baseline |         `8` | simple serving path      |
| soft router              | maybe higher/lower |         worse |         worse |         worse |               higher |        `16` | training path only       |
| hard router              |          candidate |   worse today |   worse today |   worse today |              similar |   about `8` | needs systems work       |
| hard + template batching |        same logits |        better |        better |        better |              similar |   about `8` | removes routing overhead |
| hard + template + KV     |        same logits | best expected | best expected | best expected | higher virtual slots |   about `8` | correct serving target   |

The headline should not be "router is faster" until the last row beats fixed loop on at least one realistic GPU decode setting.

The safer headline today is:

```text
Router looped transformers need route-aware serving.
The model signal is promising, and the cache/scheduler design is clear.
```

## Current Conclusion

The current router looped transformer is not yet an inference win out of the box.

The best evidence so far is more precise:

- soft router is a training mechanism, not a serving mechanism;
- hard router can slightly beat fixed-loop accuracy while keeping fixed-like block calls;
- current wall-clock latency is worse because the implementation still pays routing and grouping overhead;
- route-template replay gives `1.41x` local speedup with exact logits;
- virtual-step KV cache gives `1.27x` to `1.70x` speedup in a synthetic decoder microbench;
- vLLM and SGLang can support this class of model if cache identity includes route template and virtual route step.

The next experiment should be a small GPU serving benchmark that combines all three pieces:

```text
hard router
route-template batching
virtual-step KV cache
```

That is the real candidate inference path for router looped transformers.
