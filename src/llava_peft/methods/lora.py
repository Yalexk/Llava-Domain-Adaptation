"""LoRA fine-tuning for LLaVA.

Notes before you start:
- The config in notebooks/01_lora_scratch.ipynb (target_modules=["query", "value"],
  modules_to_save=["classifier"]) comes from a ViT image-classification example.
  LLaVA uses different module names. Print model.named_modules() to see them.
- LLaVA has two transformers: the CLIP vision tower and the Llama language model.
  Some projection names exist in both, so decide which of the two you want to adapt.
"""


def apply_lora(model, r=16, alpha=32, dropout=0.05):
	"""Wrap `model` with LoRA adapters and return the PEFT model."""
	# TODO: build a peft.LoraConfig. Things to decide:
	#   - target_modules: the Llama attention projections are q_proj/k_proj/v_proj/o_proj, but
	#     CLIP's vision tower has its own q_proj/v_proj too. A plain name list matches both.
	#     peft also accepts a regex string, e.g. one anchored on "language_model" (check the
	#     actual prefix in model.named_modules(); it differs across transformers versions).
	#   - task_type (CAUSAL_LM) and bias
	# TODO: call peft.get_peft_model(model, config)
	# TODO: print the trainable parameter count (model.print_trainable_parameters()), and
	#       return it too if you want to log it for the LoRA vs adapter comparison
	raise NotImplementedError
