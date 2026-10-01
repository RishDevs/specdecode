# SpecDecode — Project Documentation

**A speculative decoding engine for LLM inference, written from scratch in PyTorch.**
A small *draft* model proposes several tokens; a larger *target* model checks all of them in one forward pass. The point is faster single-request generation with outputs that match what the target model would produce on its own: the same tokens under greedy decoding, the same distribution under sampling.

| | |
|---|---|
| **Status** | Design stage. The repository has no code yet (as of 2026-10-01). |
| **Models** | Qwen2.5-0.5B (draft) → Qwen2.5-3B (target) |
| **Stack** | Python, PyTorch, Hugging Face Transformers + Datasets |
| **Hardware** | One NVIDIA T4 (16 GB) on Google Colab or Kaggle, free tier |
| **Cost** | $0. Every component is free. See [§8](#8-zero-cost-inventory). |

---

## Contents

1. [Scope](#1-scope)
2. [Targets and how each is measured](#2-targets-and-how-each-is-measured)
3. [How it works](#3-how-it-works)
4. [Architecture](#4-architecture)
5. [Repository layout](#5-repository-layout)
6. [Setup and running (free)](#6-setup-and-running-free)
7. [Evaluation protocol](#7-evaluation-protocol)
8. [Zero-cost inventory](#8-zero-cost-inventory)
9. [Open decisions](#9-open-decisions)
10. [Known risks and constraints](#10-known-risks-and-constraints)
11. [References](#11-references)

---

## 1. Scope

### In scope
- A draft → verify generation loop for **batch size 1**.
- **KV-cache rollback** for both models after each verification.
- **Greedy verification**: accept a draft token only if it equals the target's argmax.
- **Modified rejection sampling** (Leviathan et al., 2023; Chen et al., 2023): with temperature, top-k, or top-p, the output follows the target's distribution exactly.
- **Prompt-lookup decoding**: a drafter that needs no model. It proposes tokens by copying text that appears in the prompt.
- **Adaptive speculation length**: per request, the number of drafted tokens (γ) changes with observed acceptance. γ can drop to 0, which means plain decoding.
- **An analytical speedup model**: predicts speedup from acceptance rate and measured costs, then gets checked against measured speedup.
- **Baselines**: the project's own cached autoregressive decoder, plus Hugging Face `generate()` with `assistant_model` and with `prompt_lookup_num_tokens`.

### Out of scope
Batched or multi-request serving, tree-based speculation (Medusa, EAGLE, SpecInfer), quantization, custom CUDA kernels, training or distilling the draft model, and any web service or API.

### What "from scratch" means here
Hugging Face Transformers provides the model weights and the forward pass (`model(input_ids, past_key_values=...)`). This project writes everything above the forward pass: the decode loop, drafting, verification, the acceptance and rejection math, cache rollback, γ control, and the baseline decoder. `model.generate()` appears only as a baseline to compare against.

---

## 2. Targets and how each is measured

Each number below is a **target, not a result**. None has been measured yet. Each row names the experiment that will produce the number.

| # | Claim | Target | Measured by | Status |
|---|---|---|---|---|
| T1 | Greedy output is identical to target-only greedy | 100% of 500 prompts, fp32 | Token-by-token comparison, [§7.2](#72-exactness-greedy) | Not run |
| T2 | Sampled output follows the target distribution | TV distance < 0.02 | Empirical vs. exact distributions, [§7.3](#73-distribution-match-sampling) | Not run |
| T3 | Decode speedup with the model drafter on GSM8K and HumanEval | ≥ 1.15× at ≥ 55% acceptance | [§7.4](#74-speed) | Not run |
| T4 | Decode speedup with prompt lookup on CNN/DailyMail | ≥ 1.5× | [§7.4](#74-speed) | Not run |
| T5 | Analytical model predicts measured speedup | within 15% | [§7.5](#75-analytical-model-validation) | Not run |
| T6 | Adaptive γ prevents slowdowns on low-acceptance prompts | ≥ 1.0× | [§7.6](#76-adaptive-speculation-length) | Not run |

[§2.1](#21-feasibility-check-for-t3) shows that T3 depends on the draft/target cost ratio more than on acceptance. T1 has a numerical caveat, covered in [§10](#10-known-risks-and-constraints).

### 2.1 Feasibility check for T3

The standard model (Leviathan et al., 2023) assumes that each draft token is accepted independently with probability α, and that one verification pass costs about the same as one target decode step. Under those assumptions, the expected speedup is:

```
S(α, γ, c) = (1 − α^(γ+1)) / ((1 − α) · (γ·c + 1))
```

Here γ is the number of drafted tokens and c = (one draft step time) / (one target step time).

| α = 0.55 | γ=1 | γ=2 | γ=3 | γ=4 |
|---|---|---|---|---|
| c = 0.25 | 1.24× | 1.24× | 1.15× | 1.06× |
| c = 0.50 | 1.03× | 0.93× | 0.81× | 0.70× |

At 55% acceptance, the 1.15× target needs **c ≈ 0.25 or lower and a small γ**. The parameter-count ratio (0.5B / 3B ≈ 0.16) suggests c should be low. On a T4 in eager PyTorch, however, the 0.5B model's step time is likely dominated by Python and kernel-launch overhead, not by its size, which pushes c up. Measuring c is therefore the first benchmark to run. [§9 D3](#d3-precision-for-speed-benchmarks) and [§9 D6](#d6-reducing-draft-overhead) cover the options for lowering it.

---

## 3. How it works

### 3.1 One iteration

```
committed tokens: x_1 … x_n
1. DRAFT    drafter proposes d_1 … d_γ               (γ cheap steps, or a lookup)
2. VERIFY   target runs ONE forward pass on [x_n, d_1 … d_γ]
            → target distributions p_1 … p_{γ+1}
3. ACCEPT   walk i = 1…γ: keep d_i or reject it (rule below)
            stop at the first rejection
4. EMIT     on a rejection at i: emit a correction token drawn from p_i
            if all γ are accepted: emit a bonus token drawn from p_{γ+1}
5. ROLLBACK crop both KV caches so they cover only committed tokens
```

Each iteration commits between 1 and γ+1 tokens. Even when every draft token is rejected, the iteration still emits one token from the target, so generation always moves forward.

### 3.2 Acceptance rules

**Greedy.** Accept d_i if d_i = argmax p_i. On the first mismatch, emit argmax p_i. Each emitted token is then, by construction, the target's argmax given the previous tokens, so the output matches target-only greedy decoding as long as the logits are bitwise-stable (see [§10](#10-known-risks-and-constraints)).

**Sampling (modified rejection sampling).** Let q_i be the distribution the drafter sampled d_i from, and p_i the target's distribution. Both are processed with the same temperature, top-k, and top-p settings.
- Accept d_i with probability min(1, p_i(d_i) / q_i(d_i)).
- On rejection, sample from the residual distribution norm(max(0, p_i − q_i)).

This procedure produces samples distributed exactly as p_i. The guarantee covers the *processed* target distribution: tokens follow the target after temperature, top-k, and top-p are applied.

**Prompt lookup under sampling.** The proposal is deterministic, so q is a point mass on d_i. The rule becomes: accept with probability p_i(d_i); on rejection, sample from p_i with d_i removed and the rest renormalized. The exactness guarantee still holds.

### 3.3 KV-cache invariant

**Both caches hold keys and values for every committed token except the newest one, which goes in as the first input of the next forward pass.**

- **Target**: after accepting m drafts and emitting one more token, crop the target cache to (committed length − 1).
- **Draft**: the draft cache can end up behind the committed sequence. When all γ drafts are accepted, the draft model never saw d_γ or the bonus token. Before the next draft round, feed `committed[draft_cache_len:]`, which is one or two tokens. This is the most common off-by-one bug in speculative decoding implementations, and the cache tests target it ([§5](#5-repository-layout)).

### 3.4 Prompt-lookup drafter

Take the last *n* generated tokens (n tries from `max_ngram` down to 1). Find the most recent earlier occurrence of that n-gram in the prompt plus the output so far. Propose the γ tokens that followed it. If nothing matches, propose nothing, and the step becomes plain decoding. Cost: a scan over token ids on the CPU, with no model call, so c ≈ 0. It works well for summarization, where the output copies spans from the input.

### 3.5 Adaptive speculation length

Per request, keep a running estimate α̂ of per-token acceptance. Before each iteration, pick the γ in {0, 1, …, γ_max} that maximizes the predicted speedup S(α̂, γ, c), using measured c and verify costs. If no γ ≥ 1 is predicted to beat 1.0×, use γ = 0 (plain target decoding), and probe again with γ = 1 every N steps so the request can recover if acceptance improves. [§9 D5](#d5-adaptive-γ-policy) lists the alternatives.

**Estimating α correctly.** Positions after the first rejection are never evaluated. Computing α̂ as accepted ÷ proposed therefore underestimates α. The engine uses accepted ÷ (accepted + rejections), counting only positions that were actually evaluated.

---

## 4. Architecture

```
            ┌──────────────────────────────────────────────┐
prompt ───▶ │ Engine (engine.py)                            │ ───▶ tokens + per-step stats
            │  ├─ Drafter        ModelDrafter | PromptLookup│
            │  ├─ Verifier       target forward, γ+1 tokens │
            │  ├─ Acceptor       greedy | rejection sampling│
            │  ├─ CacheManager   crop / catch-up            │
            │  └─ GammaController fixed | adaptive          │
            └──────────────────────────────────────────────┘
                          ▲                 ▲
              models.py (load, dtype,   analytic.py (S(α,γ,c),
              vocab check)              cost fitting)
```

| Component | Responsibility | Key details |
|---|---|---|
| `models.py` | Load the draft and target on the GPU | Asserts that both tokenizers and vocab sizes match; Qwen2.5-0.5B and 3B both list 151,936 in their configs. Loads straight to the GPU to avoid running out of CPU RAM on Colab. Uses the same attention implementation (SDPA) for every run. |
| `sampling.py` | Logits → probabilities; accept/reject | One shared function applies temperature, top-k, and top-p to both p and q. Rejection sampling uses an explicit `torch.Generator` so runs are reproducible. |
| `drafters.py` | Propose γ tokens and return q | `ModelDrafter` runs γ cached steps of the 0.5B model. `PromptLookupDrafter` does an n-gram search and returns a point-mass q. |
| `cache.py` | KV rollback | Wraps the Hugging Face `DynamicCache` and its `crop()`, or a custom static cache ([§9 D2](#d2-kv-cache-implementation)). |
| `engine.py` | Main loop | Handles EOS inside an accepted block (truncates there) and `max_new_tokens` overshoot (truncates the block). Logs per-iteration γ, accepted count, and timings. |
| `baseline.py` | Target-only cached autoregressive decoding | Uses the same sampling code and RNG as the engine. This is the reference for T1 and the denominator for speedups. |
| `adaptive.py` | γ controller | Fixed, heuristic, or model-based ([§9 D5](#d5-adaptive-γ-policy)). |
| `analytic.py` | Speedup model | Fits c and the verify cost from micro-benchmarks, then predicts S for a given α and γ. |

### Prefill memory rule
During prefill, compute logits for the **last position only**. Full-prompt logits for a 1,024-token prompt are 1,024 × 151,936 × 4 bytes ≈ 620 MB in fp32, more than the free memory left on a T4 once both fp32 models are loaded. Depending on the Transformers version, use `logits_to_keep` (or the older `num_logits_to_keep`), or call the base model and apply `lm_head` to the final hidden state only.

---

## 5. Repository layout

Proposed layout; nothing exists yet.

```
specdecode/
├── specdecode/
│   ├── models.py  sampling.py  drafters.py  cache.py
│   ├── engine.py  baseline.py  adaptive.py  analytic.py
├── bench/
│   ├── data.py                 # GSM8K, HumanEval, CNN/DM prompt builders
│   ├── run_exactness.py        # T1
│   ├── run_distribution.py     # T2
│   ├── run_speed.py            # T3, T4, T6 (+ HF baselines)
│   ├── run_costs.py            # micro-benchmarks for c and verify(γ)
│   └── analyze.py              # tables, plots, T5
├── tests/                      # pytest, CPU-only, run locally
│   ├── test_rejection_sampling.py   # toy p, q over 10 tokens; 10^6 draws; TV < 0.005
│   ├── test_cache_rollback.py       # crop + catch-up equals a fresh recompute
│   ├── test_prompt_lookup.py
│   └── test_greedy_equivalence.py   # two tiny random Qwen2 models, see below
├── notebooks/run_on_colab_kaggle.ipynb
├── results/                    # JSONL per run + CSV summaries (committed)
├── requirements.txt
└── docs/
```

**Testing on CPU without a GPU.** Build two tiny Qwen2 models from a config (for example 2 layers, hidden size 64) with the real vocab size, initialized with different random seeds. They disagree often, which exercises rejection, correction, and cache rollback on a laptop in seconds. Loop bugs get caught for free before any GPU time is spent.

---

## 6. Setup and running (free)

### 6.1 Accounts (all free)
- **Google account** → Colab. Or a **Kaggle account** (phone verification is required to enable GPUs).
- **GitHub** to host the code and results.
- **Hugging Face account**: optional. The Qwen2.5 models are not gated, so downloads work without a token.

### 6.2 Local (CPU, for development and tests)
```bash
python -m venv .venv && source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
pytest tests/
```
`requirements.txt` contains `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `matplotlib`, and `pytest`. After the first working GPU run, pin exact versions with `pip freeze`. The Transformers cache and generation APIs change between releases, and an unpinned upgrade can silently break rollback.

### 6.3 GPU (Colab or Kaggle)
1. Open `notebooks/run_on_colab_kaggle.ipynb`.
2. Colab: *Runtime → Change runtime type → T4 GPU*. Kaggle: *Settings → Accelerator → GPU T4 x2* (only one GPU is used).
3. Run the setup cell: clone the repo, run `pip install -r requirements.txt`, and check that `torch.cuda.get_device_name()` reports a T4.
4. Run the experiment cells in order: `run_costs` → `run_exactness` → `run_distribution` → `run_speed` → `analyze`.
5. Save results: on Colab, write `results/` to Google Drive (free 15 GB) or push to GitHub. On Kaggle, results persist in the notebook's output, saved with the notebook version.

**Every experiment script must be resumable.** Each prompt's result is appended to a JSONL file as it finishes, and on restart the script skips prompt IDs that already have results. Free sessions disconnect.

### 6.4 Rough GPU-time budget (estimates; replace with measurements)
Decoding is limited by memory bandwidth. A T4 has about 320 GB/s, and the 3B model's fp32 weights are about 12.4 GB, so one fp32 target step takes **≥ ~39 ms** (≥ ~19 ms in fp16).

| Run | Estimate |
|---|---|
| T1 exactness: 500 prompts × 256 tokens, baseline + speculative, fp32 | ~3 h |
| T3/T4 speed: 3 datasets × ~100 prompts × 5 methods | ~3–5 h |
| Cost micro-benchmarks, T2 sampling test | < 1 h |

This fits within one Kaggle weekly GPU quota or a few Colab sessions.

---

## 7. Evaluation protocol

### 7.1 Datasets (free, on the Hugging Face Hub)
| Dataset | Hub ID | Use | Prompt |
|---|---|---|---|
| GSM8K | `openai/gsm8k` (test split) | Reasoning, model drafter | Question + "think step by step" |
| HumanEval | `openai/openai_humaneval` (164 problems) | Code, model drafter | Function signature + docstring |
| CNN/DailyMail | `abisee/cnn_dailymail` 3.0.0 (test split) | Summarization, prompt lookup | Article, truncated to ≤ 1,024 tokens, + "Summarize:" |

The 500 prompts for T1 are 200 GSM8K + 164 HumanEval + 136 CNN/DM. Prompts are chosen with a fixed seed, and the prompt IDs are saved to `results/`.

### 7.2 Exactness (greedy)
- Both runs use fp32, SDPA attention, `torch.inference_mode()`, and the same `max_new_tokens`.
- Pass condition: speculative token IDs exactly equal baseline token IDs, up to the first EOS.
- For every mismatch, log the position and the target's top-2 logit margin at that position. A mismatch at a margin below about 1e-4 is a floating-point near-tie. A mismatch at a large margin is a bug.

### 7.3 Distribution match (sampling)
Two levels:
1. **Sampler unit test.** For about 50 real (prompt, position) pairs, compute the real p and q once, then call the engine's own accept/resample function 10⁵ times with different seeds. Compare the empirical output distribution to p. This needs no extra model calls, so it is cheap.
2. **End-to-end.** For about 20 prompts, generate the first k tokens (k = 3) N times each, with both speculative and target-only sampling. Compare per-position token distributions.

**Sample size.** An empirical TV distance is biased upward by finite sampling, roughly in proportion to √(support / N). Choose N so that the TV between two target-only runs (a null baseline) is well below 0.02, and report that null value next to the measured TV.

### 7.4 Speed
- Batch size 1, a fixed `max_new_tokens`, and 3 warm-up generations that are discarded.
- Call `torch.cuda.synchronize()` around each timed region. Time prefill and decode separately.
- **Decode speedup** = baseline decode time ÷ method decode time, for the same number of output tokens. Prefill is excluded because both methods run the same prefill.
- To cancel out GPU clock and thermal drift on shared free hardware, alternate methods on each prompt rather than running all baseline prompts first.
- Report the median and mean per dataset, the acceptance rate, the mean accepted length per iteration, and the measured c.
- Methods: (a) own baseline, (b) Hugging Face `generate()` greedy with cache, (c) own speculative decoding with the model drafter, (d) Hugging Face `generate(assistant_model=draft)`, (e) own prompt lookup, (f) Hugging Face `generate(prompt_lookup_num_tokens=…)`. Set Hugging Face's assistant settings (for example `num_assistant_tokens` and its schedule) explicitly and record them, because the library defaults are adaptive and change between versions.

### 7.5 Analytical model validation
1. `run_costs.py` measures t_draft (one draft step), t_target (one target step), and t_verify(γ) for γ = 1…8.
2. Predict the speedup for each run from its measured α̂ and γ.
3. Report |predicted − measured| / measured for each dataset and each γ. Target: ≤ 15%.
4. Plot the break-even curve: for each c, the minimum α at which S > 1.

### 7.6 Adaptive speculation length
Collect the bottom acceptance quartile of prompts. Compare fixed γ = 4, adaptive γ, and the baseline on those prompts. Target: adaptive ≥ 1.0×. In practice probing costs a little, so a result slightly below 1.0× is possible. Report it as measured.

---

## 8. Zero-cost inventory

| Need | Free choice | Tradeoff vs. a paid option |
|---|---|---|
| GPU | Colab free T4 / Kaggle T4 | GPU availability is not guaranteed and sessions have time limits, so runs must be resumable. The T4 (Turing) has no bf16 tensor cores and no TF32, and it can't run FlashAttention-2 (Ampere or newer). Absolute speeds are lower than on an A100 or L4. |
| Models | Qwen2.5-0.5B (Apache-2.0), Qwen2.5-3B (Qwen Research License) | The 3B license allows non-commercial and research use only. That is fine for this project, but check the license on the model card before any commercial reuse. |
| Libraries | PyTorch, Transformers, Datasets (open source) | None |
| Datasets | GSM8K, HumanEval (MIT); CNN/DailyMail (Apache-2.0) | None |
| Experiment tracking | JSONL + pandas + matplotlib, committed to Git | No live dashboard. Weights & Biases has a free personal tier if you want one ([§9 D7](#d7-experiment-tracking)). |
| Result storage | GitHub, Google Drive (15 GB), Kaggle outputs | Size limits apply. Store results, not model weights. |
| CI for CPU tests | GitHub Actions, free for public repositories | CPU only; GPU experiments run by hand in notebooks. |

---

## 9. Open decisions

All options are free. These are your calls to make.

### D1. Base or Instruct checkpoints
| Option | Pros | Cons |
|---|---|---|
| **Instruct** (`-Instruct` for both models) | Natural prompts through the chat template; outputs are cleaner | The chat template adds tokens; acceptance may differ from base models |
| **Base** | Simpler; matches most papers | GSM8K/HumanEval outputs ramble without few-shot prompts, which lengthens runs |

Both models must use the same variant either way.

### D2. KV-cache implementation
| Option | Pros | Cons |
|---|---|---|
| **Hugging Face `DynamicCache` + `crop()`** | Least code; maintained upstream | Its API changes across Transformers versions; each crop re-slices tensors, a small overhead |
| **Custom preallocated cache with a length pointer** | Rollback just moves the pointer; makes CUDA graphs possible (D6); more "from scratch" | You must pass it into the Hugging Face model correctly (`cache_position`, attention mask); more code to test |

### D3. Precision for speed benchmarks
| Option | Pros | Cons |
|---|---|---|
| **fp32** (same as T1) | One configuration for every claim; a slower target lowers c | About 2× slower target; ~13.4 GiB of weights on a ~15 GiB T4 leaves little room ([§10](#10-known-risks-and-constraints)) |
| **fp16** | Realistic deployment speeds; plenty of memory | Greedy outputs can differ slightly from the fp16 baseline (report the match rate); c will likely be higher, which makes T3 harder; fp16 overflow needs a check on Qwen2.5 (Qwen publishes weights in bf16) |
| **Both** | Most complete | Twice the GPU time |

### D4. Speculation length bounds
- γ_max = 8 (typical) or 4 (fewer wasted draft steps when acceptance is low). Prompt lookup can use a higher γ_max (for example 10) because its drafts cost almost nothing.

### D5. Adaptive γ policy
| Option | Pros | Cons |
|---|---|---|
| **Model-based** (§3.5: argmax of S(α̂, γ, c), fall back to γ=0) | Uses the analytical model directly; a principled γ=0 fallback | Depends on accurate c; needs a probe interval |
| **Heuristic** (+2 after full acceptance, −1 otherwise; what Hugging Face does) | Simple and robust | Never reaches γ=0, so low-acceptance prompts can still be slower than the baseline |
| **Draft-confidence stop** (stop drafting when the draft's max probability falls below a threshold) | Adapts within a single iteration | Adds a threshold to tune; doesn't fit the i.i.d. analytical model as cleanly |

### D6. Reducing draft overhead
| Option | Pros | Cons |
|---|---|---|
| **None (eager PyTorch)** | Simplest; honest baseline | c is probably high on a T4 |
| **`torch.compile`** on the draft | Large speedups are possible for small models | Compile time on every session; static shapes needed; Triton support on T4 (sm_75) should be tested |
| **CUDA graphs** for the draft step | Removes per-kernel launch overhead | Needs the static cache (D2-custom) and fixed shapes |

### D7. Experiment tracking
JSONL + matplotlib (no account needed, fully offline) or Weights & Biases free tier (live dashboards; needs an account and depends on an external service).

### D8. Analytical model form
| Option | Pros | Cons |
|---|---|---|
| **i.i.d. α** (Leviathan) | Closed form; standard; easy to defend | Acceptance is correlated across positions in practice, which biases predictions |
| **Empirical accepted-length distribution** | Captures that correlation | Less general; needs data for each dataset |

Recommendation: report both, and use the gap between them as a finding.

---

## 10. Known risks and constraints

1. **fp32 memory on a T4.** Weights: 3B ≈ 12.4 GB (11.5 GiB) and 0.5B ≈ 2.0 GB (1.8 GiB), about 13.4 GiB together. Colab's T4 shows about 15 GiB, and the CUDA context takes a few hundred MiB. KV cache: about 72 KiB per token (3B) plus 24 KiB per token (0.5B). This fits only if prefill computes last-position logits only, prompts are capped (about 1,024 tokens), and everything runs under `inference_mode`.
2. **CPU RAM when loading.** Colab free has about 12–13 GB of system RAM. Loading the 3B model in fp32 through the CPU can run out of memory. Load with `device_map="cuda"` and `torch_dtype=torch.float32` so shards stream onto the GPU.
3. **"100% identical" is a numerical claim, not just a logical one.** The verify pass runs γ+1 tokens through matrix multiplications with different shapes than one-token decoding. Different kernels can sum in a different order, so logits can differ in the last bits. When the top two logits are nearly tied, the argmax can flip. fp32 makes this rare but cannot rule it out. The T1 protocol therefore logs the margin for every mismatch ([§7.2](#72-exactness-greedy)). If any mismatch occurs, report it with its margin rather than dropping the prompt.
4. **Upcasting is not extra accuracy.** The published weights are bf16. Running in fp32 reduces numerical divergence between the baseline and verify paths. It does not make the model better.
5. **Draft-step overhead** may make c too high for T3. This is the first thing to measure ([§2.1](#21-feasibility-check-for-t3)).
6. **Free-tier variability.** GPU clocks and availability vary between sessions. Report the hardware details (`nvidia-smi`) with each run, and compare methods only within one session.
7. **Transformers API drift.** Cache classes, `cache_position`, and logits-trimming arguments have changed across releases. Pin versions, and keep `test_cache_rollback.py` in CI.

---

## 11. References
- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
- Saxena. *Prompt Lookup Decoding.* GitHub, 2023.
- Hugging Face Transformers docs: assisted generation and prompt lookup decoding (`generate()` arguments).
- Qwen2.5 model cards on the Hugging Face Hub (architecture configs and licenses).
