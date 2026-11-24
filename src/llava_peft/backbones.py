"""Per-backbone settings, so every script works on LLaVA or Qwen3-VL by changing --model-id.

Everything model-specific lives here: which class loads the weights, which layer the
"projector" method trains, and how images are sized before the processor sees them.
"""

from dataclasses import dataclass

from PIL import ImageOps
from transformers import LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration


@dataclass(frozen=True)
class Backbone:
	model_class: type
	# Full-match regex for the layer the "projector" method unfreezes. None = not chosen yet.
	projector_target: str | None
	# Side in pixels every image is center-cropped to. None = leave sizing to the processor.
	image_size: int | None


BACKBONES = {
	"llava-hf/llava-1.5-7b-hf": Backbone(
		model_class=LlavaForConditionalGeneration,
		projector_target=r".*multi_modal_projector\.linear_2",
		# The CLIP processor already crops to 336, so every image is 576 tokens
		image_size=None,
	),
	"Qwen/Qwen3-VL-2B-Instruct": Backbone(
		model_class=Qwen3VLForConditionalGeneration,
		# TODO(owner): Qwen3-VL has no single vision->LM bridge. Options:
		#   r".*visual\.merger\.linear_fc2"  main merger only, 4096->2048, 8.39M params
		#   or also the 3 DeepStack mergers (visual.deepstack_merger_list.N), which feed
		#   ViT layers 5/11/17 into the first decoder layers. apply_projector_tuning
		#   currently expects the regex to match exactly one module.
		projector_target=None,
		# Dynamic resolution: the token count follows the image size (81-300 on CUB-sized
		# images), which would break the collator's cached prompt length.
		# 448 / (patch 16 * merge 2) = 14 -> 28x28 patches -> 196 image tokens, always.
		image_size=448,
	),
}


def get_backbone(model_id):
	if model_id not in BACKBONES:
		raise KeyError(f"No backbone settings for '{model_id}'. Add it to BACKBONES in "
			f"backbones.py. Known: {list(BACKBONES)}")
	return BACKBONES[model_id]


def prepare_image(image, size=None):
	"""Convert to RGB and, if size is set, resize the short side and center-crop to size x size.

	Cropping rather than squashing keeps aspect ratios, matching what LLaVA's CLIP processor does.
	"""
	image = image.convert("RGB")
	if size is not None:
		image = ImageOps.fit(image, (size, size))
	return image
