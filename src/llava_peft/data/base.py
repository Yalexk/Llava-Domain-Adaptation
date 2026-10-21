from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

from datasets import load_dataset


@dataclass
class DatasetSpec:
    name: str
    hf_id: str
    prompt: str
    # Converts a raw HF row into {"image": PIL.Image, "label": str}
    to_example: Callable[[dict, "DatasetSpec"], dict]
    # Splits as named on the hub, e.g. {"train": "train", "test": "test"}
    splits: dict = field(default_factory=lambda: {"train": "train", "test": "test"})

    def load_raw(self, split):
        return load_dataset(self.hf_id, split=self.splits[split])

    def load(self, split, limit=None):
        ds = self.load_raw(split)
        if limit is not None:
            ds = ds.shuffle(seed=0).select(range(min(limit, len(ds))))
        return ds.map(lambda row: self.to_example(row, self), remove_columns=ds.column_names)

    @cached_property
    def class_names(self):
        raise NotImplementedError
