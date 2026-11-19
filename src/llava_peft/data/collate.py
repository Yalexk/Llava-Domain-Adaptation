"""Turn {image, label} examples into training batches (input_ids, attention_mask, pixel_values, labels).

The key difference from generation: the model sees prompt + answer, but the loss
should only be computed on the answer tokens.
"""

from llava_peft.model import make_messages


class TrainCollator:
	"""Callable for a DataLoader / Trainer: list of {image, label} dicts -> batch dict of tensors."""

	def __init__(self, processor, prompt):
		self.processor = processor
		# "USER: <image>\n<prompt> ASSISTANT:" (same string generate() feeds the model)
		self.prompt_text = processor.apply_chat_template(make_messages(prompt), add_generation_prompt=True)
		self._prompt_len = None

	def _prompt_length(self, image):
		# Token count of the prompt alone, *after* the processor expands <image> into its
		# patch tokens (576 for llava-1.5). The prompt and image-token count are the same for
		# every example, so compute it once.
		if self._prompt_len is None:
			ids = self.processor(images=image.convert("RGB"), text=self.prompt_text, return_tensors="pt")["input_ids"]
			self._prompt_len = ids.shape[1]
		return self._prompt_len

	def __call__(self, examples):
		eos = self.processor.tokenizer.eos_token
		# Full training text = prompt + " label</s>". The explicit EOS teaches the model to stop,
		# so at inference it emits the species name and nothing more.
		texts = [f"{self.prompt_text} {ex['label']}{eos}" for ex in examples]
		images = [ex["image"].convert("RGB") for ex in examples]

		batch = self.processor(images=images, text=texts, padding=True, return_tensors="pt")

		input_ids = batch["input_ids"]
		attention_mask = batch["attention_mask"]
		seq_len = input_ids.shape[1]
		prompt_len = self._prompt_length(images[0])

		labels = input_ids.clone()
		labels[attention_mask == 0] = -100  # padding

		# Mask the prompt (incl. image tokens). Where it sits depends on the padding side:
		# right padding -> starts at column 0; left padding -> after that row's pad tokens.
		left = self.processor.tokenizer.padding_side == "left"
		for i in range(len(examples)):
			start = seq_len - int(attention_mask[i].sum()) if left else 0
			labels[i, start:start + prompt_len] = -100

		batch["labels"] = labels
		return batch
