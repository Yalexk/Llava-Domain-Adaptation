"""Scoring free-text predictions against fine-grained class labels.

Some options, roughly strictest to loosest:
  exact_match     normalized prediction == normalized label
  contains_match  label appears inside a longer answer ("This is a kentucky warbler.")
  top1 / top5     map each prediction to its nearest class name (string or embedding
                  similarity) and score it as closed-set classification accuracy
  cosine_sim      embedding similarity between prediction and label

A caution about cosine similarity on its own: in fine-grained domains wrong answers
embed close to right ones ("rusty blackbird" vs "red winged blackbird"), so a wrong
prediction can still score high. Worth testing that on a few pairs before relying on it.
"""

import re


def normalize(text):
	"""Normalize the string for equivalent outputs"""
	if text is None:
		return ""
	s = str(text).lower().strip()
	s = s.removeprefix("a photo of a ")
	s = s.replace("-", " ")
	s = re.sub(r"[^a-z0-9 ]", " ", s)
	s = re.sub(r"\b(a|an|the)\b", " ", s)
	s = " ".join(s.split())
	return s


def exact_match(pred, label):
	"""True if pred and label are identical after normalizing."""
	return normalize(pred) == normalize(label)


def contains_match(pred, label):
	"""True if the normalized label appears inside the normalized pred."""
	return normalize(label) in normalize(pred)


def score_predictions(preds, labels, class_names):
	"""Return (per_example: list[dict], summary: dict) for parallel lists of strings."""
	per_example = []
	exacts = 0
	contains = 0
	# strict=True raises if the lists differ in length instead of silently truncating
	for pred, label in zip(preds, labels, strict=True):
		exact = exact_match(pred, label)
		contain = contains_match(pred, label)
		exacts += exact
		contains += contain
		per_example.append({"pred": pred, "label": label, "exact_match": exact, "contains_match": contain})

	n = len(per_example)
	if n == 0:
		return per_example, {"n": 0}

	summary = {"n": n, "exact_match": exacts / n, "contains_match": contains / n}
	return per_example, summary
