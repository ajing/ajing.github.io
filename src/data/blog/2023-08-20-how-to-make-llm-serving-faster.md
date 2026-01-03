---
author: Jing Lu
pubDatetime: 2023-08-20T00:00:00Z
title: "How to Make LLM Inference Faster"
featured: false
draft: false
tags:
  - LLM
  - Inference
  - Performance
  - ML
description: "An overview of LLM inference optimization techniques including KV cache, FlashAttention, and memory management strategies."
---

## An overview of LLM inference

You begin with a sequence of tokens referred to as the "prefix" or "prompt". At this stage, the model doesn't need to do anything. Then, the LLM generates a sequence of completion tokens, continuing until it reaches a stop token or the maximum sequence length.

One word is composed of multiple tokens, depending on the tokenization method. It would require a few iterations to obtain a single word. Because of the iterative nature, it's hard to parallel the text generation process. Usually the tokens do not map 1:1 to ASCII characters. One popular token encoding technique is Byte-Pair Encoding (BPE), which is a subword tokenization technique that breaks down words into smaller subword units. A subword unit in BPE can be represented by 2 to 4 bytes.

![Simplified LLM inference.](/images/2208/llm_inference.png "image_tooltip")

The model starts with the prompt (yellow) and generates one token at a time (blue) until it reaches the end-of-sequence token. This is done for each input sequence in the batch. The prompt part of computation can run in parallel.

LLM inference is considered to be memory-I/O bound, not compute bound. This means that the time it takes to load the data into memory is more important than the time it takes to process the data. Only when the batch size is large enough will the compute time take longer than the I/O time. So, if the batch is small, caching (i.e. kv cache) is not necessary.

To be more specific, on A100 GPU with 40GB RAM, according to [6], 13B parameter models, taking 26GB RAM for storing model parameters, consumes nearly 1MB of state for each token in a sequence. 40-26=14GB are left for processing sequences. ~14K tokens can be held in memory at once. If the sequence length is 2048, our batch size is limited to 7 sequences. Clearly, if you self-host your own large language model and you are the only user, you will not get much from batch processing.

With this memory constraint, we have two options to speed up: enhance the speed of a single run or improve the speed of independent runs.

---

## Improve speed of a single run

### KV Cache

For a LLM task on a GPT architecture, we can reduce the dimensionality of the attention matrix computation by focusing on the new attention of the last token in each pass. The last token does not influence the intermediate embedding calculation of all preceding tokens due to masked multi-head attention. We don't need to compute all embeddings for key and value for each new token predicted. Even prior to receiving any user input, it is entirely feasible to precompute all these embeddings. However, this approach does not confer any memory-saving benefits, as the precomputed embeddings persist in occupying memory space.

![Simplified LLM inference.](/images/2208/precompute.png "image_tooltip")

Tokenizer is also a big factor to improve the latency. The HuggingFace tokenizers package uses the Rust implementation of the model tokenizer in combination with smart caching to achieve a speedup of up to 10x for overall latency.

### FlashAttention

When the FlashAttention paper came out, GPT-3.5-turbo could handle a context length of 16K, while GPT-4 could handle about 33K. FlashAttention makes training faster (by 3 times) and lets models handle longer sequences. The token length by using this method is up to 16K. While the original method used memory based on the square of the sequence length, FlashAttention uses memory in direct proportion to the sequence length.

The FlashAttention paper is inspired by differences in memory types. HBM has about 1,000 times more capacity than SRAM, but SRAM is only about 10 times faster. So, it's more efficient to store data in SRAM, do a lot of processing there, and only move data to DRAM when necessary.

![original](/images/2208/original.png "image_tooltip")

![tiling](/images/2208/tiling.png "image_tooltip")

### FlashAttention v2

---

## References

1. [HuggingFace Blog - Accelerated Inference](https://huggingface.co/blog/accelerated-inference#getting-to-the-first-10x-speedup)
2. How continuous batching enables 23x throughput in LLM inference while reducing p50 latency
3. [vLLM](https://vllm.ai/)
4. [HuggingFace - GPU Inference](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
5. [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/#kv-cache)
6. [LLM Numbers - GPU Memory Requirements](https://github.com/ray-project/llm-numbers#1-mb-gpu-memory-required-for-1-token-of-output-with-a-13b-parameter-model)
7. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

