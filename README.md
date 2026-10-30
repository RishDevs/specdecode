<img src=".github/header.svg" width="100%" alt="specdecode — SpecDecode — README header">

# SpecDecode

**A speculative decoding engine for LLM inference, written from scratch in PyTorch.**

A small *draft* model proposes several tokens, and a larger *target* model checks all of them in a single forward pass. The goal is faster single-request generation with **no change to the output**: greedy decoding produces exactly the tokens the target model would produce on its own, and sampling follows the target model's distribution exactly.

> **Status:** in active development. Benchmarks and results will be added here once experiments are complete.

---

## Why speculative decoding?

When an LLM generates one token at a time, the GPU spends most of each step reading the model's weights from memory, not computing. Checking several tokens in one pass costs about the same as generating one. If a cheap draft model guesses correctly often enough, the target model can commit several tokens per pass instead of one.

## How it works

```
1. DRAFT     the draft model proposes γ tokens: d₁ … d_γ
2. VERIFY    the target model scores all γ tokens in ONE forward pass
3. ACCEPT    keep draft tokens left to right until the first rejection
4. EMIT      add one token from the target (a correction, or a bonus if all γ were accepted)
5. ROLLBACK  trim both KV caches back to the committed tokens, then repeat
```

- **Greedy mode:** a draft token is kept only if it matches the target's top choice.
- **Sampling mode:** modified rejection sampling (Leviathan et al., 2023; Chen et al., 2023) guarantees the output follows the target model's distribution exactly.

## Features

- [ ] Draft → verify loop for batch size 1
- [ ] KV-cache rollback for the draft and target models
- [ ] Greedy verification and modified rejection sampling (temperature / top-k / top-p)
- [ ] Prompt-lookup decoding: a drafter that needs no model and copies n-grams from the prompt
- [ ] Adaptive speculation length that falls back to plain decoding when speculation doesn't pay off
- [ ] Analytical speedup model based on acceptance rate and the draft/target cost ratio
- [ ] Benchmarks against cached autoregressive decoding and Hugging Face assisted generation

## Stack

| | |
|---|---|
| Models | [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (draft) → [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) (target) |
| Libraries | PyTorch, Hugging Face Transformers, Datasets |
| Hardware | One NVIDIA T4 (free tier on Google Colab or Kaggle) |
| Benchmarks | GSM8K, HumanEval, CNN/DailyMail |

Everything in this project runs at **zero cost**: open-source models and libraries, public datasets, and free-tier GPUs.

## Getting started

> Code is being added. These steps describe the intended workflow.

**Local (CPU, for development and tests):**

```bash
git clone https://github.com/RishDevs/specdecode.git
cd specdecode
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

**GPU (free):** open the notebook in `notebooks/` on Google Colab (*Runtime → Change runtime type → T4 GPU*) or on Kaggle (*Accelerator → GPU T4*), then run the cells in order.

## Documentation

See [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) for the full design: architecture, KV-cache invariants, the evaluation protocol, and open design decisions.

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
- Saxena. *Prompt Lookup Decoding.* 2023.

## License

The code is released under the [MIT License](LICENSE).

Model weights are not included in this repository; they are downloaded from Hugging Face under their own licenses. Qwen2.5-0.5B is Apache-2.0. Qwen2.5-3B is under the Qwen Research License, which allows non-commercial use only.
