"""Turn {image, label} examples into training batches (input_ids, attention_mask, pixel_values, labels).

The key difference from generation: the model sees prompt + answer, but the loss
should only be computed on the answer tokens.
"""

from llava_peft.backbones import prepare_image
from llava_peft.model import make_messages


class TrainCollator:
	"""Callable for a DataLoader / Trainer: list of {image, label} dicts -> batch dict of tensors."""

	def __init__(self, processor, prompt, image_size=None):
		self.processor = processor
		self.image_size = image_size
		# "USER: <image>\n<prompt> ASSISTANT:" (same string generate() feeds the model)
		self.prompt_text = processor.apply_chat_template(make_messages(prompt), add_generation_prompt=True)
		self._prompt_len = None
		self._grid = None

	def _prompt_length(self, image):
		# Token count of the prompt alone, *after* the processor expands <image> into its
		# patch tokens (576 for llava-1.5, 196 for Qwen3-VL at 448px). Constant only because
		# every image has a fixed size, so compute it once. __call__ checks that assumption.
		if self._prompt_len is None:
			out = self.processor(images=image, text=self.prompt_text, return_tensors="pt")
			self._prompt_len = out["input_ids"].shape[1]
			# Qwen-style processors report each image's patch grid; LLaVA's doesn't
			self._grid = out.get("image_grid_thw")
		return self._prompt_len

	def __call__(self, examples):
		eos = self.processor.tokenizer.eos_token
		# Full training text = prompt + " label</s>". The explicit EOS teaches the model to stop,
		# so at inference it emits the species name and nothing more.
		texts = [f"{self.prompt_text} {ex['label']}{eos}" for ex in examples]
		images = [prepare_image(ex["image"], self.image_size) for ex in examples]

		batch = self.processor(images=images, text=texts, padding=True, return_tensors="pt")
		prompt_len = self._prompt_length(images[0])

		# A different patch grid means a different image-token count, so the cached prompt
		# length would mask the wrong span without any error. Fail loudly instead.
		grid = batch.get("image_grid_thw")
		if grid is not None and not (grid == self._grid[0]).all():
			raise ValueError(f"image token grids differ ({grid.tolist()} vs {self._grid.tolist()}): "
				"pass a fixed image_size so every image becomes the same number of tokens")

		input_ids = batch["input_ids"]
		attention_mask = batch["attention_mask"]
		seq_len = input_ids.shape[1]

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
