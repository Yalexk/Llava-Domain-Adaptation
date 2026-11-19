"""Fine-tune LLaVA with a PEFT method and save the adapter weights.

	python scripts/train.py --dataset birds --method lora --run-name birds_lora_r16 --limit 50

Outputs in results/<run-name>/ (gitignored):
  adapter/        saved PEFT weights (load with scripts/evaluate.py --adapter-path)
  train_metrics.json   config + trainable params + wall-clock + peak memory + loss history
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from llava_peft.data import get_dataset
from llava_peft.data.collate import TrainCollator
from llava_peft.methods.lora import apply_lora
from llava_peft.model import load_model
from llava_peft.utils import get_device, peak_memory_gb, setup_env

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METHODS = {"lora": apply_lora}


def sync(device):
	# CUDA calls return before the GPU finishes, so wait for it before reading the clock
	if device == "cuda":
		torch.cuda.synchronize()


def pick_dtype(device):
	# bf16 has fp32's range, so no loss scaling is needed. T4s lack it and fall back to fp16.
	if device == "cuda" and torch.cuda.is_bf16_supported():
		return torch.bfloat16
	return torch.float16


@torch.no_grad()
def validation_loss(model, loader, device, dtype):
	"""Mean per-batch loss over the validation loader."""
	model.eval()
	losses = []
	for batch in loader:
		batch = batch.to(device, dtype)
		losses.append(model(**batch).loss.item())
	model.train()
	return sum(losses) / len(losses)


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
	parser.add_argument("--warmup-ratio", type=float, default=0.03)
	parser.add_argument("--max-grad-norm", type=float, default=1.0)
	parser.add_argument("--rank", type=int, default=16)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--log-every", type=int, default=10, help="optimizer steps between loss log entries")
	parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
		help="recompute activations in backward to save memory (needed for batch 4 on a 24GB GPU)")
	parser.add_argument("--run-name", required=True)
	args = parser.parse_args()

	setup_env()
	device = get_device()
	dtype = pick_dtype(device)
	print(f"device: {device}, dtype: {dtype}")

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model, processor = load_model(args.model_id, device=device, dtype=dtype)
	model = METHODS[args.method](model, r=args.rank)
	if args.grad_checkpointing:
		# Non-reentrant checkpointing works with a frozen embedding layer, so
		# enable_input_require_grads() isn't needed.
		model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
	model.train()

	spec = get_dataset(args.dataset)
	ds = spec.load("train", limit=args.limit)
	if args.val_fraction > 0:
		split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)
		train_ds, val_ds = split["train"], split["test"]
	else:
		train_ds, val_ds = ds, None
	print(f"train: {len(train_ds)} examples, val: {len(val_ds) if val_ds else 0}")

	collator = TrainCollator(processor, spec.prompt)
	generator = torch.Generator().manual_seed(args.seed)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
		num_workers=args.num_workers, generator=generator)
	val_loader = None
	if val_ds is not None:
		val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers)

	params = [p for p in model.parameters() if p.requires_grad]
	trainable_params = sum(p.numel() for p in params)
	optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
	steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
	total_steps = steps_per_epoch * args.epochs
	scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.warmup_ratio * total_steps), total_steps)
	# Only fp16 needs loss scaling (its small range underflows gradients). Disabled otherwise.
	scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))

	# Reset so peak memory covers training only, not model loading
	if device == "cuda":
		torch.cuda.reset_peak_memory_stats()
	sync(device)
	start = time.perf_counter()
	val_seconds = 0.0

	loss_history, val_history = [], []
	step = 0
	window = []  # micro-batch losses since the last optimizer step
	progress = tqdm(total=total_steps, desc="training")
	for epoch in range(args.epochs):
		for i, batch in enumerate(train_loader):
			batch = batch.to(device, dtype)
			loss = model(**batch).loss
			scaler.scale(loss / args.grad_accum).backward()
			window.append(loss.item())

			if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()
				optimizer.zero_grad(set_to_none=True)
				step += 1

				mean_loss = sum(window) / len(window)
				window.clear()
				progress.update(1)
				progress.set_postfix(loss=f"{mean_loss:.4f}")
				if step % args.log_every == 0 or step == total_steps:
					loss_history.append({"step": step, "loss": mean_loss, "lr": scheduler.get_last_lr()[0]})

		if val_loader is not None:
			sync(device)
			t0 = time.perf_counter()
			val_loss = validation_loss(model, val_loader, device, dtype)
			sync(device)
			val_seconds += time.perf_counter() - t0
			val_history.append({"epoch": epoch + 1, "step": step, "val_loss": val_loss})
			print(f"epoch {epoch + 1}: val loss {val_loss:.4f}")
	progress.close()

	sync(device)
	# Training time excludes validation so it's comparable across runs with different val sizes
	seconds = time.perf_counter() - start - val_seconds

	out_dir = RESULTS_DIR / args.run_name
	out_dir.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(out_dir / "adapter")

	metrics = {
		"config": vars(args),
		"device": device,
		"dtype": str(dtype),
		"gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
		"train_examples": len(train_ds),
		"val_examples": len(val_ds) if val_ds is not None else 0,
		"trainable_params": trainable_params,
		"optimizer_steps": step,
		"train_seconds": seconds,
		"peak_memory_gb": peak_memory_gb(device),
		"loss_history": loss_history,
		"val_history": val_history,
	}
	with open(out_dir / "train_metrics.json", "w") as f:
		json.dump(metrics, f, indent=2)
	print(f"trainable params: {trainable_params:,}, train time: {seconds:.1f}s")
	print(f"saved to {out_dir}")


if __name__ == "__main__":
	main()
