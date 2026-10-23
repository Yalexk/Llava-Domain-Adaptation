from functools import cached_property

from .base import DatasetSpec


class CarsSpec(DatasetSpec):
	@cached_property
	def class_names(self):
		return list(self.load_raw("train").features["label"].names)


CARS = CarsSpec(
	name="cars",
	hf_id="tanganke/stanford_cars",
	prompt="What is the make, model, and year of the car in this photo? "
		   "Answer with only the make, model, and year.",
	# Class names look like "Acura TL Sedan 2012"
	to_example=lambda row, spec: {"image": row["image"], "label": spec.class_names[row["label"]]},
)
