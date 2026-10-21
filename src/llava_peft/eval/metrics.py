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


def normalize(text):
    """Lowercase, strip punctuation, articles, and the "a photo of a" prefix."""
    # TODO
    raise NotImplementedError


def score_predictions(preds, labels, class_names):
    """Return (per_example: list[dict], summary: dict) for parallel lists of strings."""
    # TODO
    raise NotImplementedError
