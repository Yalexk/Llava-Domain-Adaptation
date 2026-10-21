# TODO

Time estimates are rough, for someone learning as they go. Debugging and reading docs are included. Total ≈ 35–55 hours.

## Phase 1: Zero-shot baseline (birds) · ~4–6h

- [ ] `model.py`: implement `load_model` (LlavaForConditionalGeneration + AutoProcessor, left padding) · 30m
- [ ] `model.py`: implement batched `generate` and return only the new tokens · 1h
- [ ] Sanity-check on ~20 birds images in a notebook and look at what the model actually says · 30m
- [ ] Tweak the prompt (`data/birds.py`) until answers are short species names · 30m
- [ ] `eval/metrics.py`: `normalize`, exact match, contains match · 45m
- [ ] Test cosine similarity on a few hand-picked right/wrong pairs before trusting it · 30m
- [ ] `eval/metrics.py`: nearest-class mapping → top-1 / top-5 accuracy · 1h
- [ ] `scripts/evaluate.py`: wire it together, save predictions + metrics + timing · 1h
- [ ] Run zero-shot on the full birds test set on the GPU and record the numbers · 30m

## Phase 2: Remote GPU setup · ~1–2h (do alongside phase 1)

- [ ] Clone the repo on the GPU box, create the venv, copy `.env`, set `HF_HOME` · 30m
- [ ] Confirm `get_device()` prints `cuda` and the model loads in fp16 · 15m
- [ ] Get comfortable with tmux + rsync for results · 30m

## Phase 3: LoRA on birds · ~8–12h

- [ ] Read the LoRA paper's method section + PEFT LoRA docs · 1h
- [ ] Inspect `model.named_modules()`, decide which layers to target (language model only? vision tower too?) · 30m
- [ ] Fix `methods/lora.py` (the notebook config is from a ViT example) · 30m
- [ ] Build training data: chat template with prompt → label as the answer, and mask the prompt tokens in the loss (`labels = -100`) · 2h
- [ ] `scripts/train.py`: training loop (HF `Trainer` or plain PyTorch). Log loss, wall-clock, peak memory, trainable params · 3h
- [ ] Overfit ~50 examples first to prove the loop works · 1h
- [ ] Full training run, save adapter weights, evaluate with `--adapter-path` · 1–2h (plus GPU time)
- [ ] Hold out a validation split from train so you're not tuning on test · 30m

## Phase 4: Adapter method on birds · ~8–12h

- [ ] Choose what "adapter" means and write down why · 1–2h reading
  - Bottleneck adapters (Houlsby / Pfeiffer) in the language model
  - LLaMA-Adapter (learnable prompts + zero-init gating)
  - Projector-only tuning (what LLaVA itself trains in stage 1)
- [ ] Implement in `methods/adapter.py` · 3–5h
- [ ] Same training script, same data, same eval · 1h
- [ ] Try to match trainable-param counts with LoRA for a fair comparison · 1h

## Phase 5: Stanford Cars · ~3–5h

- [ ] Zero-shot eval on cars (should only need `--dataset cars`) · 1h
- [ ] LoRA + adapter runs on cars · 2–4h (mostly GPU time)

## Phase 6: Analysis + write-up · ~6–10h

- [ ] Results table: method × domain × {top-1, top-5, train time, peak mem, trainable params}
- [ ] Ablations if time: LoRA rank, number of training examples, which layers
- [ ] Error analysis: which classes get confused (confusion pairs)
- [ ] Write up findings in the README or a report

## Nice-to-have

- [ ] Fix seeds and log configs per run (or use wandb)
- [ ] Try a smaller VLM for fast iteration on the Mac
