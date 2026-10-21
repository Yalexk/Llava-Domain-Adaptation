"""Dataset registry.

Every dataset is exposed through the same interface so the rest of the code
doesn't care which domain it's running on:

    spec = get_dataset("birds")
    ds = spec.load("test")          # HF Dataset with columns: image, label
    spec.class_names                # sorted list of all label strings
    spec.prompt                     # question asked to the model
"""

from .birds import BIRDS
from .cars import CARS

DATASETS = {spec.name: spec for spec in (BIRDS, CARS)}


def get_dataset(name):
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Options: {list(DATASETS)}")
    return DATASETS[name]
