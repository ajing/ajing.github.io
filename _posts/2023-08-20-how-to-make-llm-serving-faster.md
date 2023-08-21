# How to make LLM inference faster?

## An overview of LLM inference

You begin with a sequence of tokens referred to as the “prefix” or “prompt”.  At this stage, the model doesn’t need to do anything. Then, the LLM generates a sequence of completion tokens, continuing until it reaches a stop token or the maximum sequence length.

One word is composed of multiple tokens, depending on the tokenization method. It would require a few iterations to obtain a single word. Because of the iterative nature, it’s hard to parallel the text generation process. Usually the tokens do not map 1:1 to ASCII characters. One popular token encoding technique is Byte-Pair Encoding (BPE), which is a subword tokenization technique that breaks down words into smaller subword units. A subword unit in BPE can be represented by 2 to 4 bytes.


![Simplified LLM inference.](/images/2208/llm_inference.png "image_tooltip")


The model starts with the prompt (yellow) and generates one token at a time (blue) until it reaches the end-of-sequence token. This is done for each input sequence in the batch. The prompt part of computation can run in parallel.

LLM inference is considered to be memory-I/O bound, not compute bound. This means that the time it takes to load the data into memory is more important than the time it takes to process the data. Only when the batch size is large enough will the compute time take longer than the I/O time. So, if the batch is small, caching (i.e. kv cache) is not necessary.

To be more specific, on A100 GPU with 40GB RAM, according to [6], 13B parameter models, taking 26GB RAM for storing model parameters, consumes nearly 1MB of state for each token in a sequence.  40-26=14GB are left for processing sequences. ~14K tokens can be held in memory at once. If the sequence length is 2048, our batch size is limited to 7 sequences. Clearly, if you self-host your own large language model and you are the only user, you will not get much from batch processing.

With this memory constraint, we have two options to speed up: enhance the speed of a single run or improve the speed of independent runs.


## Improve speed of a single run


### kv cache

For a LLM task on a GPT architecture, we can reduce the dimensionality of the attention matrix computation by focusing on the new attention of the last token in each pass. The last token does not influence the intermediate embedding calculation of all preceding tokens due to masked multi-head attention. We don’t need to compute all embeddings for key and value for each new token predicted. Even prior to receiving any user input, it is entirely feasible to precompute all these embeddings. However, this approach does not confer any memory-saving benefits, as the precomputed embeddings persist in occupying memory space.


![Simplified LLM inference.](/images/2208/precompute.png "image_tooltip")


### Tokenizer

The HuggingFace tokenizers package uses the Rust implementation of the model tokenizer in combination with smart caching to achieve a speedup of up to 10x for overall latency.

Reference:
[1]https://huggingface.co/blog/accelerated-inference#getting-to-the-first-10x-speedup
[2]How continuous batching enables 23x throughput in LLM inference while reducing p50 latency
[3]https://vllm.ai/
[4]https://huggingface.co/docs/transformers/perf_infer_gpu_one
[5]https://kipp.ly/blog/transformer-inference-arithmetic/#kv-cache
[6]https://github.com/ray-project/llm-numbers#1-mb-gpu-memory-required-for-1-token-of-output-with-a-13b-parameter-model