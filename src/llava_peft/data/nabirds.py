"""NABirds: 48,562 photos of North American birds (23,929 train / 24,633 test, the official split).

The hub copy labels each image with one of 555 *visual categories*: a species plus a plumage
variant, e.g. "Red-winged Blackbird (Female/juvenile)". Species with a single variant carry an
export artefact instead, e.g. "Great Blue Heron (ID: 480)". Both are collapsed to the species name
(404 classes), so the task matches CUB: name the species, not its sex or age.
"""

import re
from functools import cached_property

from datasets import load_dataset_builder

from .base import DatasetSpec
from .birds import BIRDS

# A trailing "(...)" holds the variant or the "ID: n" artefact
VARIANT = re.compile(r"\s*\([^)]*\)\s*$")


def species_from_category(name):
	"""'Blue-winged Teal  (Male)' -> 'Blue-winged Teal'. Also collapses stray double spaces."""
	return re.sub(r"\s+", " ", VARIANT.sub("", name)).strip()


class NABirdsSpec(DatasetSpec):
	@cached_property
	def category_names(self):
		# The 555 raw ClassLabel names. Read from the hub metadata, so this downloads no images.
		return load_dataset_builder(self.hf_id).info.features[self.label_column].names

	@cached_property
	def category_to_species(self):
		return [species_from_category(n) for n in self.category_names]

	@cached_property
	def class_names(self):
		return sorted(set(self.category_to_species))


NABIRDS = NABirdsSpec(
	name="nabirds",
	hf_id="zguo0525/nabirds-dataset",
	# Same question as CUB, so the two bird datasets are comparable
	prompt=BIRDS.prompt,
	label_column="label",
	to_label=lambda i, spec: spec.category_to_species[i],
)
