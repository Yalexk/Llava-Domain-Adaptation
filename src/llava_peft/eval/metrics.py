"""Scoring free-text predictions against fine-grained class labels.

Some options, roughly strictest to loosest:
  exact_match     normalized prediction == normalized label
  contains_match  label appears inside a longer answer ("This is a kentucky warbler.")
  top1            map each prediction to its nearest class name by embedding similarity
                  and score it as closed-set classification accuracy

Why not raw cosine between pred and label (see notebooks/test_similarity.ipynb): wrong
answers embed close to right ones, e.g. "red bellied woodpecker" vs "red headed woodpecker"
scored 0.91 while "Northern Cardinal" vs "cardinal" scored 0.77, so no threshold works.
Ranking against every class name does work: a close-but-wrong answer is itself a class
name, so it maps to itself and is scored wrong.

Top-5 from this ranking is left out on purpose: it ranks by name similarity, so a wrong
species in the right family ("rusty blackbird") gets top-5 credit almost every time.
"""

import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


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


@lru_cache
def load_embedder(model_id=EMBED_MODEL_ID):
	# Cached so repeated scoring calls don't reload the model
	return SentenceTransformer(model_id)


def nearest_classes(preds, class_names):
	"""Map each pred to its most similar class name.

	Returns one dict per pred:
	  nearest      the top-ranked class name
	  top1_score   its cosine similarity (about 1.0 when pred is already a class name)
	  gap          top1_score minus the runner-up's score; a small gap means a vague
	               answer like "warbler" that fits many classes about equally well
	"""
	if len(preds) == 0:
		return []

	embedder = load_embedder()
	# Unit-length vectors, so a dot product is the cosine similarity
	class_emb = embedder.encode(list(class_names), normalize_embeddings=True)  # (n_classes, dim)
	pred_emb = embedder.encode(list(preds), normalize_embeddings=True)  # (n_preds, dim)
	scores = pred_emb @ class_emb.T  # (n_preds, n_classes)

	# argsort is ascending, so reverse each row to put the best match first
	order = np.argsort(scores, axis=1)[:, ::-1]

	results = []
	for row, ranked in zip(scores, order):
		results.append({
			"nearest": class_names[ranked[0]],
			"top1_score": float(row[ranked[0]]),
			"gap": float(row[ranked[0]] - row[ranked[1]]),
		})
	return results


def score_predictions(preds, labels, class_names):
	"""Return (per_example: list[dict], summary: dict) for parallel lists of strings."""
	preds, labels = list(preds), list(labels)
	if len(preds) == 0:
		return [], {"n": 0}

	nearest = nearest_classes(preds, class_names)

	per_example = []
	exacts = 0
	contains = 0
	top1s = 0
	# strict=True raises if the lists differ in length instead of silently truncating
	for pred, label, near in zip(preds, labels, nearest, strict=True):
		exact = exact_match(pred, label)
		contain = contains_match(pred, label)
		top1 = near["nearest"] == label
		exacts += exact
		contains += contain
		top1s += top1
		per_example.append({
			"pred": pred,
			"label": label,
			"exact_match": exact,
			"contains_match": contain,
			"top1": top1,
			**near,
		})

	n = len(per_example)
	summary = {"n": n, "exact_match": exacts / n, "contains_match": contains / n, "top1": top1s / n}
	return per_example, summary
