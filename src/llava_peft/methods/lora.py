"""LoRA fine-tuning for VLMs (LLaVA, Qwen3-VL)."""

from peft import LoraConfig, get_peft_model


# Default target. peft does a full-match on this regex against each module's name. Anchoring on
# "language_model" skips CLIP's vision tower, which also has q_proj/k_proj/v_proj.
# Other placements (--location) pass their own target from backbones.py.
# (Names in this transformers version: model.language_model.layers.N.self_attn.q_proj,
# for both LLaVA and Qwen3-VL. Qwen's vision blocks use attn.qkv/attn.proj, so never match.)
LM_ATTENTION_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"


def apply_lora(model, r=16, alpha=32, dropout=0.05, target=LM_ATTENTION_REGEX):
	"""Wrap every Linear full-matching `target` with LoRA and return the PEFT model.

	The target is saved in the adapter's adapter_config.json, so PeftModel.from_pretrained
	reloads the same placement without being told.
	"""
	config = LoraConfig(
		r=r,
		lora_alpha=alpha,
		lora_dropout=dropout,
		target_modules=target,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, config)
	# Trainable count for the LoRA vs adapter comparison
	model.print_trainable_parameters()
	return model
