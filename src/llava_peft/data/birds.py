from functools import cached_property

from .base import DatasetSpec

PREFIX = "a photo of a "


def label_from_text(text):
	# Labels look like "A photo of a kentucky warbler"
	text = text.strip()
	if text.lower().startswith(PREFIX):
		text = text[len(PREFIX):]
	return text.lower()


class BirdsSpec(DatasetSpec):
	@cached_property
	def class_names(self):
		# Selecting the text column avoids decoding every image
		return sorted({label_from_text(t) for t in self.load_raw("train")["text"]})

# Test prompts
"""
What species of bird is in this photo? Answer with only the species name.
What is the exact species of the bird in this photo? Give its full common name as a birdwatcher would, not just the general type of bird.
Identify the bird species in this photo. Reply with its full common name only, for example Great Blue Heron.
This photo shows one of 200 North American bird species. Which species is it? Give the full common name, including descriptive words such as color or markings.
"""

BIRDS = BirdsSpec(
	name="birds",
	hf_id="cassiekang/cub200_dataset",
	prompt="Identify the bird species in this photo. Reply with its full common name only, for example Great Blue Heron.",
	label_column="text",
	to_label=lambda text, spec: label_from_text(text),
)
