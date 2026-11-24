from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable

from datasets import load_dataset


@dataclass
class DatasetSpec:
	name: str
	hf_id: str
	prompt: str
	# The raw column holding the label, and how to turn one value of it into the label string.
	# Labels are built from this column alone, so loading never decodes an image.
	label_column: str
	to_label: Callable[[Any, "DatasetSpec"], str]
	image_column: str = "image"
	# Splits as named on the hub, e.g. {"train": "train", "test": "test"}
	splits: dict = field(default_factory=lambda: {"train": "train", "test": "test"})

	def load_raw(self, split):
		return load_dataset(self.hf_id, split=self.splits[split])

	def load(self, split, limit=None):
		"""Return an HF Dataset with columns image (PIL, decoded on access) and label (str)."""
		ds = self.load_raw(split)
		if limit is not None:
			ds = ds.shuffle(seed=0).select(range(min(limit, len(ds))))
		labels = [self.to_label(value, self) for value in ds[self.label_column]]
		# Keep the image column as stored instead of map()-ing every row: map would decode each
		# image and re-encode it into a new cache file, which is slow on big datasets, doubles
		# disk use, and re-compresses JPEGs.
		ds = ds.select_columns([self.image_column])
		if self.image_column != "image":
			ds = ds.rename_column(self.image_column, "image")
		return ds.add_column("label", labels)

	@cached_property
	def class_names(self):
		raise NotImplementedError
