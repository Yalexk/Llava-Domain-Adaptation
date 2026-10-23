import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def load_model(model_id=DEFAULT_MODEL_ID, device="cpu", dtype=torch.float16):
    """Return (model, processor)."""
    # The processor bundles the image preprocessor (resize/normalize for CLIP)
    # and the tokenizer, so one object prepares both halves of the input.
    processor = AutoProcessor.from_pretrained(model_id)

    # generate() appends new tokens to the right end of each sequence. With right
    # padding, shorter prompts would have pad tokens between the prompt and the answer.
    processor.tokenizer.padding_side = "left"

    model = LlavaForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device)
    return model, processor


def make_messages(prompt):
    # {"type": "image"} is a placeholder. apply_chat_template turns it into an
    # <image> token, and the processor fills that in with the actual image features.
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]


@torch.inference_mode()
def generate(model, processor, images, prompt, max_new_tokens=30):
    """Run a batch of PIL images through the model with one prompt and return a list of strings.

    Return only the newly generated text, without the prompt.
    """
    # 1. Messages -> prompt string, e.g. "USER: <image>\n<prompt> ASSISTANT:"
    text = processor.apply_chat_template(make_messages(prompt), add_generation_prompt=True)

    # 2. Images + strings -> tensors (input_ids, attention_mask, pixel_values).
    #    padding=True pads every prompt to the same length so they stack into one batch.
    #    .to(device, dtype) only casts float tensors (pixel_values), not the token ids.
    inputs = processor(
        images=[img.convert("RGB") for img in images],
        text=[text] * len(images),
        padding=True,
        return_tensors="pt",
    ).to(model.device, model.dtype)

    # 3. Greedy decoding (do_sample=False) so results are deterministic and comparable
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # 4. The output is prompt + answer. Every row is padded to the same prompt length,
    #    so slicing off the first input_ids.shape[1] tokens leaves only the answer.
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return [s.strip() for s in processor.batch_decode(new_tokens, skip_special_tokens=True)]
