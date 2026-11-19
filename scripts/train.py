"""Fine-tune LLaVA with a PEFT method and save the adapter weights.

	python scripts/train.py --dataset birds --method lora --run-name birds_lora_r16 --limit 50

Outputs in results/<run-name>/ (gitignored):
  adapter/        saved PEFT weights (load with scripts/evaluate.py --adapter-path)
  train_metrics.json   config + trainable params + wall-clock + peak memory + loss history
"""

import argparse
import json
import time
from pathlib import Path

import torch

from llava_peft.data import get_dataset
from llava_peft.data.collate import TrainCollator
from llava_peft.methods.lora import apply_lora
from llava_peft.model import load_model
from llava_peft.utils import get_device, peak_memory_gb, setup_env

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METHODS = {"lora": apply_lora}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="birds")
	parser.add_argument("--method", default="lora", choices=list(METHODS))
	parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
	parser.add_argument("--limit", type=int, default=None, help="train on a random subset (use ~50 to overfit-test)")
	parser.add_argument("--val-fraction", type=float, default=0.1, help="held out from train, not test")
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--grad-accum", type=int, default=4)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--rank", type=int, default=16)
	parser.add_argument("--run-name", required=True)
	args = parser.parse_args()

	setup_env()
	device = get_device()
	print(f"device: {device}")

	# TODO: set seeds (torch.manual_seed etc.) so runs are comparable

	model, processor = load_model(args.model_id, device=device)
	model = METHODS[args.method](model, r=args.rank)

	spec = get_dataset(args.dataset)
	train_ds = spec.load("train", limit=args.limit)
	# TODO: split off a validation set from train_ds (datasets has .train_test_split)

	collator = TrainCollator(processor, spec.prompt)

	# TODO: choose HF Trainer (TrainingArguments + Trainer) or a plain PyTorch loop.
	#   Things either way needs to handle:
	#   - fp16 weights + trainable LoRA params: consider autocast / fp32 for the adapter params
	#     and why training pure fp16 weights with AdamW can go wrong (T4 has no bf16)
	#   - gradient accumulation, LR schedule + warmup
	#   - model.gradient_checkpointing_enable() + enable_input_require_grads() if memory is tight
	#   - remove_unused_columns=False if using Trainer (images aren't model inputs until collated)
	#   - evaluating loss on the val split every N steps

	# Reset so peak memory covers training only, not model loading
	if device == "cuda":
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.synchronize()
	start = time.perf_counter()

	# TODO: run training here

	if device == "cuda":
		torch.cuda.synchronize()
	seconds = time.perf_counter() - start

	out_dir = RESULTS_DIR / args.run_name
	out_dir.mkdir(parents=True, exist_ok=True)

	# TODO: model.save_pretrained(out_dir / "adapter")

	# TODO: count trainable params (sum p.numel() for p in model.parameters() if p.requires_grad)
	metrics = {
		"config": vars(args),
		"device": device,
		"gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
		"trainable_params": None,
		"train_seconds": seconds,
		"peak_memory_gb": peak_memory_gb(device),
		"loss_history": [],
	}
	with open(out_dir / "train_metrics.json", "w") as f:
		json.dump(metrics, f, indent=2)
	print(f"saved to {out_dir}")


if __name__ == "__main__":
	main()
