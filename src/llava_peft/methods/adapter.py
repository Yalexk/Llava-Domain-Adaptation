"""Non-LoRA PEFT methods for VLMs, written by hand (peft has no bottleneck adapters).

- bottleneck: Pfeiffer-style adapter after the MLP of every language-model decoder layer.
- projector:  train only the vision->text bridge layer named by the backbone (backbones.py).
              LLaVA: `multi_modal_projector.linear_2`, the 4096x4096 bridge.

Both keep the trainable weights in fp32 even when the base model runs in bf16/fp16.
(peft does the same for LoRA.) Updating bf16 weights directly loses small steps to rounding,
and GradScaler refuses to unscale fp16 gradients.
"""

import json
import re
from pathlib import Path

import torch
from torch import nn

# Full-match on the module name, e.g. model.language_model.layers.7.mlp
LM_MLP_REGEX = re.compile(r".*language_model.*\.layers\.\d+\.mlp")
# Default for checkpoints saved before the target was recorded in custom_adapter.json (LLaVA runs)
DEFAULT_PROJECTOR_TARGET = r".*multi_modal_projector\.linear_2"

WEIGHTS_FILE = "trainable.pt"
CONFIG_FILE = "custom_adapter.json"


class BottleneckAdapter(nn.Module):
	"""x + up(act(down(x))), with `up` zero-initialised so the adapter starts as the identity."""

	def __init__(self, hidden_size, bottleneck):
		super().__init__()
		self.down = nn.Linear(hidden_size, bottleneck)
		self.act = nn.GELU()
		self.up = nn.Linear(bottleneck, hidden_size)
		# Zero `up` -> the model begins training exactly equal to the frozen base model
		nn.init.zeros_(self.up.weight)
		nn.init.zeros_(self.up.bias)

	def forward(self, x):
		h = self.up(self.act(self.down(x.float())))
		return x + h.to(x.dtype)


class MLPWithAdapter(nn.Module):
	"""Frozen MLP followed by an adapter. The decoder layer then adds the residual as usual."""

	def __init__(self, mlp, hidden_size, bottleneck):
		super().__init__()
		self.mlp = mlp
		self.adapter = BottleneckAdapter(hidden_size, bottleneck)

	def forward(self, x):
		return self.adapter(self.mlp(x))


class FP32Linear(nn.Module):
	"""Holds a copy of a Linear in fp32 and casts activations in and out around it."""

	def __init__(self, linear):
		super().__init__()
		self.module = linear.float()

	def forward(self, x):
		return self.module(x.float()).to(x.dtype)


def _replace(model, name, new_module):
	parent_name, _, child = name.rpartition(".")
	setattr(model.get_submodule(parent_name), child, new_module)


def _freeze(model):
	for p in model.parameters():
		p.requires_grad = False


def _report(model):
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	total = sum(p.numel() for p in model.parameters())
	print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")


def apply_bottleneck_adapter(model, bottleneck=64):
	"""Freeze `model`, insert a bottleneck adapter after every language-model MLP, return the model.

	Params: 32 layers x (2 * 4096 * bottleneck + 4096 + bottleneck). bottleneck=64 -> 16.9M,
	close to LoRA r=16 on q/k/v/o (16.78M).
	"""
	_freeze(model)
	# Collect first: replacing modules while iterating named_modules() is unsafe
	targets = [n for n, _ in model.named_modules() if LM_MLP_REGEX.fullmatch(n)]
	if not targets:
		raise ValueError("no language-model MLP modules matched LM_MLP_REGEX; check model.named_modules()")
	hidden = model.config.text_config.hidden_size
	for name in targets:
		mlp = model.get_submodule(name)
		device = next(mlp.parameters()).device
		wrapped = MLPWithAdapter(mlp, hidden, bottleneck)
		# Only the new adapter is fp32/trainable. The wrapped MLP keeps its frozen dtype.
		wrapped.adapter.to(device=device, dtype=torch.float32)
		_replace(model, name, wrapped)
	_report(model)
	return model


def apply_projector_tuning(model, target=DEFAULT_PROJECTOR_TARGET):
	"""Freeze `model` and unfreeze only the module whose name full-matches `target`.

	LLaVA's linear_2: 4096*4096 + 4096 = 16.78M params.
	"""
	if target is None:
		raise ValueError("no projector target set for this backbone; choose one in backbones.py")
	_freeze(model)
	targets = [n for n, _ in model.named_modules() if re.fullmatch(target, n)]
	if len(targets) != 1:
		raise ValueError(f"expected exactly one module matching {target!r}, found {targets}")
	wrapped = FP32Linear(model.get_submodule(targets[0]))
	for p in wrapped.parameters():
		p.requires_grad = True
	_replace(model, targets[0], wrapped)
	_report(model)
	return model


APPLY = {"bottleneck": apply_bottleneck_adapter, "projector": apply_projector_tuning}


def save_trainable(model, path, method, **hparams):
	"""Save only the trainable tensors plus what's needed to rebuild the wrapper structure."""
	path = Path(path)
	path.mkdir(parents=True, exist_ok=True)
	state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
	torch.save(state, path / WEIGHTS_FILE)
	with open(path / CONFIG_FILE, "w") as f:
		json.dump({"method": method, **hparams}, f, indent=2)


def is_custom_adapter(path):
	return (Path(path) / CONFIG_FILE).exists()


def load_trainable(model, path):
	"""Re-apply the saved method to a fresh base model, then load the saved tensors into it."""
	path = Path(path)
	with open(path / CONFIG_FILE) as f:
		config = json.load(f)
	method = config.pop("method")
	model = APPLY[method](model, **config)
	state = torch.load(path / WEIGHTS_FILE, map_location="cpu")
	# strict=False because the state holds only the trainable tensors, so check the keys by hand
	result = model.load_state_dict(state, strict=False)
	if result.unexpected_keys:
		raise ValueError(f"unexpected keys in {path / WEIGHTS_FILE}: {result.unexpected_keys[:5]}")
	missing_trainable = {n for n, p in model.named_parameters() if p.requires_grad} - set(state)
	if missing_trainable:
		raise ValueError(f"trainable params missing from checkpoint: {sorted(missing_trainable)[:5]}")
	return model
