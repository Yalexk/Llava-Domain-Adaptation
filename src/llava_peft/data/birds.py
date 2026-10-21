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


BIRDS = BirdsSpec(
    name="birds",
    hf_id="cassiekang/cub200_dataset",
    prompt="What species of bird is in this photo? Answer with only the species name.",
    to_example=lambda row, spec: {"image": row["image"], "label": label_from_text(row["text"])},
)
