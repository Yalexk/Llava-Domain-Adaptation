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
    # TODO: build a peft.LoraConfig and call get_peft_model
    # TODO: print the trainable parameter count (useful for the LoRA vs adapter comparison)
    raise NotImplementedError
