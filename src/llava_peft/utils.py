import os
import warnings

import torch


def setup_env():
	"""Load .env, log in to Hugging Face, and quiet library logging."""
	from dotenv import load_dotenv
	from huggingface_hub import login
	from huggingface_hub.utils import disable_progress_bars
	import transformers

	load_dotenv()
	os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

	transformers.logging.set_verbosity_error()
	warnings.filterwarnings("ignore")
	disable_progress_bars()

	token = os.getenv("HF_TOKEN")
	if token:
		login(token=token)


def get_device():
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def peak_memory_gb(device):
	if device == "cuda":
		return torch.cuda.max_memory_allocated() / 1e9
	if device == "mps":
		return torch.mps.driver_allocated_memory() / 1e9
	return None
