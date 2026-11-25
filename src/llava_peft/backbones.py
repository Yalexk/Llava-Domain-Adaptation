"""Per-backbone settings, so every script works on LLaVA or Qwen3-VL by changing --model-id.

Everything model-specific lives here: which class loads the weights, where each PEFT method
can be placed (--location) and at what size, and how images are sized before the processor.
"""

from dataclasses import dataclass, field

from PIL import ImageOps
from transformers import LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration

# lm_mlp / vision_mlp are the sublayer control: LoRA on the MLP instead of attention, LoRA only
LOCATIONS = ["lm", "projector", "vision", "lm_mlp", "vision_mlp"]


@dataclass(frozen=True)
class Location:
	"""Where in the model a method goes, and the sizes that give every cell the same budget.

	Regexes are full-matched against module names. None = that method isn't defined here.
	"""
	# LoRA wraps every Linear matching this
	lora_target: str | None
	lora_rank: int | None
	# A bottleneck adapter goes after every module matching this ...
	adapter_target: str | None
	# ... sized adapter_hidden -> adapter_size -> adapter_hidden (the matched modules' output width)
	adapter_hidden: int | None
	adapter_size: int | None


@dataclass(frozen=True)
class Backbone:
	model_class: type
	# Full-match regex for the layer the "projector" method (full fine-tune) unfreezes. None = unset.
	projector_target: str | None
	# Side in pixels every image is center-cropped to. None = leave sizing to the processor.
	image_size: int | None
	locations: dict[str, Location] = field(default_factory=dict)


LM_ATTENTION = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"
LM_MLP = r".*language_model.*\.layers\.\d+\.mlp"
# The Linears inside each LM MLP, for LoRA (LM_MLP above is the whole block, for the adapter)
LM_MLP_LINEARS = r".*language_model.*\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)"


BACKBONES = {
	"llava-hf/llava-1.5-7b-hf": Backbone(
		model_class=LlavaForConditionalGeneration,
		projector_target=r".*multi_modal_projector\.linear_2",
		# The CLIP processor already crops to 336, so every image is 576 tokens
		image_size=None,
		# The published runs, all ~16.8M: LoRA 16,777,216 / adapter 16,910,336 / projector 16,781,312
		locations={
			"lm": Location(LM_ATTENTION, 16, LM_MLP, 4096, 64),
			# Adapter on the projector output: 2*4096*2047 + 2047 + 4096 = 16,775,167. No projector
			# LoRA: 16.8M on linear_1 (1024 inputs) would need r~1260, above its full rank of 1024.
			"projector": Location(None, None, r".*multi_modal_projector", 4096, 2047),
		},
	),
	"Qwen/Qwen3-VL-2B-Instruct": Backbone(
		model_class=Qwen3VLForConditionalGeneration,
		# TODO(owner): full fine-tune target. Qwen3-VL has no single vision->LM bridge:
		#   r".*visual\.merger\.linear_fc2"  main merger only, 4096->2048, 8.39M params
		#   or also the 3 DeepStack mergers (visual.deepstack_merger_list.N).
		projector_target=None,
		# Dynamic resolution: the token count follows the image size (81-300 on CUB-sized
		# images), which would break the collator's cached prompt length.
		# 448 / (patch 16 * merge 2) = 14 -> 28x28 patches -> 196 image tokens, always.
		image_size=448,
		# Every cell ~1.6M trainable params, anchored on LoRA r=4 in the LM (1,605,632)
		locations={
			# LoRA q/k/v/o r=4: 28 * 4 * (4096 + 3072 + 3072 + 4096) = 1,605,632
			# adapter after each MLP m=13: 28 * (2*2048*13 + 13 + 2048) = 1,548,652 (-3.5%)
			"lm": Location(LM_ATTENTION, 4, LM_MLP, 2048, 13),
			# Main merger only: neither pattern can match deepstack_merger_list.N, so the 3 DeepStack
			# paths reach the first decoder layers without passing through the trained part.
			# LoRA fc1+fc2 r=112: 112 * ((4096+4096) + (4096+2048)) = 1,605,632
			# adapter on the merger output m=391: 2*2048*391 + 391 + 2048 = 1,603,975 (-0.1%)
			"projector": Location(r".*visual\.merger\.(linear_fc1|linear_fc2)", 112,
				r".*visual\.merger", 2048, 391),
			# 24 ViT blocks at 1024-d, before the merger. Edits here also reach the DeepStack paths.
			# LoRA qkv+proj r=11: 24 * 11 * ((1024+3072) + (1024+1024)) = 1,622,016 (+1.0%)
			# adapter after each ViT MLP m=32: 24 * (2*1024*32 + 32 + 1024) = 1,598,208 (-0.5%)
			"vision": Location(r".*visual\.blocks\.\d+\.attn\.(qkv|proj)", 11,
				r".*visual\.blocks\.\d+\.mlp", 1024, 32),
			# Sublayer control: LoRA on the MLP Linears instead of attention, so LoRA vs adapter can be
			# separated from attention vs MLP (the adapter already sits on the MLP). LoRA only.
			# LM gate/up/down r=2: 28 * 2 * 3 * (2048 + 6144) = 1,376,256 (-14%; r=3 would be +29%)
			"lm_mlp": Location(LM_MLP_LINEARS, 2, None, None, None),
			# ViT fc1+fc2 r=7: 24 * 7 * 2 * (1024 + 4096) = 1,720,320 (+7%; r=6 would be -8%)
			"vision_mlp": Location(r".*visual\.blocks\.\d+\.mlp\.(linear_fc1|linear_fc2)", 7,
				None, None, None),
		},
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
