"""Generate predictions on a dataset split and score them.

    python scripts/evaluate.py --dataset birds --limit 200 --run-name birds_zeroshot

Suggested outputs in results/<run-name>/ (gitignored):
  predictions.jsonl   {"pred", "label"} per example; save these so you can re-score
                      without running the model again
  metrics.json        config + metric summary + timing (seconds/example, peak memory)
"""

import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="birds")
	parser.add_argument("--split", default="test")
	parser.add_argument("--limit", type=int, default=None, help="random subset size")
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
	parser.add_argument("--adapter-path", default=None, help="saved PEFT weights to load")
	parser.add_argument("--run-name", required=True)
	args = parser.parse_args()

	# TODO: setup_env(), get_device()
	# TODO: load model (+ PeftModel.from_pretrained if --adapter-path)
	# TODO: spec = get_dataset(args.dataset); ds = spec.load(args.split, limit=args.limit)
	# TODO: batch over ds -> generate(...), time it
	# TODO: score_predictions(...) and write the outputs above


if __name__ == "__main__":
	main()
