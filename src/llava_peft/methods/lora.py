"""LoRA fine-tuning for LLaVA."""

from peft import LoraConfig, get_peft_model


# peft does a full-match on this regex against each module's name. Anchoring on
# "language_model" skips CLIP's vision tower, which also has q_proj/k_proj/v_proj.
# (Names in this transformers version: model.language_model.layers.N.self_attn.q_proj)
LM_ATTENTION_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"


def apply_lora(model, r=16, alpha=32, dropout=0.05):
	"""Wrap `model` with LoRA adapters and return the PEFT model."""
	config = LoraConfig(
		r=r,
		lora_alpha=alpha,
		lora_dropout=dropout,
		target_modules=LM_ATTENTION_REGEX,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, config)
	# Trainable count for the LoRA vs adapter comparison
	model.print_trainable_parameters()
	return model
