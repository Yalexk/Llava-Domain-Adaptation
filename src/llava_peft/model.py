import torch

DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def load_model(model_id=DEFAULT_MODEL_ID, device="cpu", dtype=torch.float16):
    """Return (model, processor).

    Hint: LlavaForConditionalGeneration + AutoProcessor. For batched generation the
    tokenizer has to pad on the left.
    """
    # TODO
    raise NotImplementedError


def generate(model, processor, images, prompt, max_new_tokens=30):
    """Run a batch of PIL images through the model with one prompt and return a list of strings.

    Return only the newly generated text, without the prompt.
    """
    # TODO
    raise NotImplementedError
