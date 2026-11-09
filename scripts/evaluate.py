"""Generate predictions on a dataset split and score them.

	python scripts/evaluate.py --dataset birds --limit 200 --run-name birds_zeroshot

Outputs in results/<run-name>/ (gitignored):
  predictions.jsonl   one line per example: pred, label, nearest class, gap, ambiguous, ...
                      raw preds are written before scoring so a crash doesn't cost a model rerun
  metrics.json        config + metric summary + timing (seconds/example, peak memory)
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from llava_peft.data import get_dataset
from llava_peft.eval.metrics import score_predictions
from llava_peft.model import generate, load_model
from llava_peft.utils import get_device, peak_memory_gb, setup_env

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def sync(device):
	# CUDA calls return before the GPU finishes, so wait for it before reading the clock
	if device == "cuda":
		torch.cuda.synchronize()


def run_generation(model, processor, ds, prompt, batch_size, device):
	"""Return (preds, labels, seconds) for the whole dataset."""
	preds, labels = [], []
	sync(device)
	start = time.perf_counter()
	for i in tqdm(range(0, len(ds), batch_size), desc="generating"):
		# Slicing an HF Dataset gives a dict of columns: {"image": [...], "label": [...]}
		batch = ds[i:i + batch_size]
		preds.extend(generate(model, processor, batch["image"], prompt))
		labels.extend(batch["label"])
	sync(device)
	return preds, labels, time.perf_counter() - start


def write_jsonl(path, rows):
	with open(path, "w") as f:
		for row in rows:
			f.write(json.dumps(row) + "\n")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="birds")
	parser.add_argument("--split", default="test")
	parser.add_argument("--limit", type=int, default=None, help="random subset size (seeded)")
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
	parser.add_argument("--adapter-path", default=None, help="saved PEFT weights to load")
	parser.add_argument("--run-name", required=True)
	args = parser.parse_args()

	setup_env()
	device = get_device()
	print(f"device: {device}")

	model, processor = load_model(args.model_id, device=device)
	if args.adapter_path:
		from peft import PeftModel
		model = PeftModel.from_pretrained(model, args.adapter_path)
	model.eval()

	spec = get_dataset(args.dataset)
	ds = spec.load(args.split, limit=args.limit)
	print(f"{args.dataset}/{args.split}: {len(ds)} examples, {len(spec.class_names)} classes")

	# Reset after loading so peak memory covers inference only
	if device == "cuda":
		torch.cuda.reset_peak_memory_stats()

	preds, labels, seconds = run_generation(model, processor, ds, spec.prompt, args.batch_size, device)
	peak_gb = peak_memory_gb(device)

	out_dir = RESULTS_DIR / args.run_name
	out_dir.mkdir(parents=True, exist_ok=True)
	pred_path = out_dir / "predictions.jsonl"

	# Save raw predictions first, in case scoring fails
	write_jsonl(pred_path, ({"pred": p, "label": l} for p, l in zip(preds, labels, strict=True)))

	per_example, summary = score_predictions(preds, labels, spec.class_names)

	# Overwrite with the scored rows (adds nearest, gap, ambiguous, ...) for error analysis
	write_jsonl(pred_path, per_example)

	metrics = {
		"config": vars(args),
		"device": device,
		"metrics": summary,
		"timing": {
			"total_seconds": seconds,
			"seconds_per_example": seconds / len(ds) if len(ds) else None,
			# On MPS this is current driver allocation, not a true peak; report CUDA numbers
			"peak_memory_gb": peak_gb,
		},
	}
	with open(out_dir / "metrics.json", "w") as f:
		json.dump(metrics, f, indent=2)

	print(json.dumps(metrics["metrics"], indent=2))
	print(f"saved to {out_dir}")


if __name__ == "__main__":
	main()
