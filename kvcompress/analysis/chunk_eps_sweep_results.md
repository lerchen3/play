## Perplexity & Compression Summary (AIME, 5 prompts, max 256 tokens) - run_whitened_chunk

- Baseline (full cache): ppl **1.1653**
- KVMerger 0.8 cosine threshold:
  - ppl 1.9472
  - mean compression 0.6325
- QFilters:
  - ppl 2.0383
  - mean compression 0.4291
- ε = 10.0 (no projection):
  - ppl 1.6451
  - mean compression 0.7957
- ε = 15.0 (no projection):
  - ppl 2.3089
  - mean compression 0.4591
- ε = 20.0 (no projection):
  - ppl 1.9567
  - mean compression 0.3052
TODO: run the actual lse least squares approach

TODO: rerun with lots more epsilons and more prompts. we're probably just seeing noise rn
