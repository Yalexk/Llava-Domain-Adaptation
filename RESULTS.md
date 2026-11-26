# Results

Results so far. All runs: 2 epochs, lr 2e-4, effective batch 16, seed 0, one run per cell. Raw metrics are in `reports/`.

**Top-1** is closed-set accuracy: each answer is mapped to its nearest class name. **Exact** means the answer is a class name exactly as written.

## Qwen3-VL-2B on NABirds (404 species, 24,633 test images)

Trained on an L4 in bf16, except runs marked † (Kaggle T4 in fp16). Zero-shot scores **31.4%** top-1 and 24.4% exact.

### Method × location at a matched budget (~1.6M trainable params)

| Top-1 (exact) | Vision encoder | Projector (merger) | Language model |
|---|---|---|---|
| **LoRA** | 76.1% (75.7%) | **76.7%** (76.6%) | 75.3% (75.3%) |
| **Adapter** | 70.8% (70.5%) | 72.2% (72.0%) | 75.2% (75.1%) |

### Same-rank control: LoRA at r=4 everywhere

| Location | r=4 params | r=4 top-1 | Matched-budget top-1 | Gap |
|---|---|---|---|---|
| Language model | 1,605,632 | 75.3% | 75.3% (r=4, same run) | 0 |
| Vision encoder | 589,824 | 65.7% | 76.1% (r=11) | −10.4 |
| Projector | 57,344 | 52.2% | 76.7% (r=112) | −24.5 |

### Sublayer: LoRA on attention vs MLP, at a matched budget

| Location | Attention (top-1) | MLP (top-1) | Difference |
|---|---|---|---|
| Language model | 75.3%: q/k/v/o, r=4, 1,605,632 params | **79.6%**†: gate/up/down, r=2, 1,376,256 params | +4.3 |
| Vision encoder | 76.1%: qkv/proj, r=11, 1,622,016 params | 74.5%†: fc1/fc2, r=7, 1,720,320 params | −1.6 |

### All runs

| Run | Trainable params | Top-1 | Exact | Ambiguous | Val loss (ep 1 → 2) | Train time | Train peak mem | Eval s/img |
|---|---|---|---|---|---|---|---|---|
| zero-shot | 0 | 31.4% | 24.4% | 25.4% | – | – | – | 0.119 |
| `lora_lm` (r=4) | 1,605,632 | 75.3% | 75.3% | 0.7% | 0.198 → 0.148 | 1.49 h | 6.2 GB | 0.149 |
| `adapter_lm` (m=13) | 1,548,652 | 75.2% | 75.1% | 0.7% | 0.201 → 0.143 | 1.33 h | 6.2 GB | 0.123 |
| `lora_projector` (r=112) | 1,605,632 | 76.7% | 76.6% | 1.0% | 0.218 → 0.153 | 1.25 h | 6.2 GB | 0.116 |
| `lora_projector_r4` | 57,344 | 52.2% | 51.3% | 4.3% | 0.480 → 0.438 | 1.26 h | 6.2 GB | 0.117 |
| `adapter_projector` (m=391) | 1,603,975 | 72.2% | 72.0% | 1.2% | 0.252 → 0.189 | 1.24 h | 6.2 GB | 0.116 |
| `lora_vision` (r=11) | 1,622,016 | 76.1% | 75.7% | 2.4% | 0.289 → 0.202 | 2.08 h | 6.4 GB | 0.125 |
| `lora_vision_r4` | 589,824 | 65.7% | 64.9% | 4.0% | 0.389 → 0.307 | 2.10 h | 6.4 GB | 0.125 |
| `adapter_vision` (m=32) | 1,598,208 | 70.8% | 70.5% | 2.4% | 0.322 → 0.243 | 1.86 h | 6.4 GB | 0.116 |
| `lora_lm_mlp` (r=2)† | 1,376,256 | 79.6% | 79.5% | 0.6% | 0.161 → 0.117 | 3.68 h | 6.2 GB | 0.233 |
| `lora_vision_mlp` (r=7)† | 1,720,320 | 74.5% | 74.3% | 2.1% | 0.273 → 0.203 | 5.41 h | 6.4 GB | 0.222 |

**Ambiguous** is the share of answers whose closest and second-closest class names are within 0.07 of each other. Eval peak memory is 4.5 GB for every run.

† Trained and evaluated on a Kaggle T4 in fp16, so it isn't a like-for-like comparison with the L4 runs. Their train time and eval speed are slower because of the GPU, not the method.

## LLaVA-1.5-7B on CUB-200 (200 species, 5,794 test images)

First round, before the location grid. Trained on an L4 in bf16 with ~17M trainable params each.

| Method | Trainable params | Top-1 | Exact | Train time | Train peak mem | Eval s/img |
|---|---|---|---|---|---|---|
| LoRA (r=16) | 16,777,216 | 73.6% | 73.5% | 2.53 h | 16.5 GB | 0.375 |
| Bottleneck adapter (m=64) | 16,910,336 | 72.0% | 72.0% | 2.22 h | 16.5 GB | 0.340 |
| Projector (full fine-tune of last linear) | 16,781,312 | **83.3%** | 83.2% | 2.08 h | 16.5 GB | 0.332 |

On LLaVA, tuning the projector beat LoRA by about 10 points, which is what led to the location grid on Qwen.
