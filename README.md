# Llava-domain-adaptation

This project investigates how well parameter-efficient fine-tuning (PEFT) adapts multimodal language models to different visual domains. Using LLaVA as the baseline, it evaluates LoRA and adapter-based PEFT on fine-grained datasets, comparing performance and compute cost across methods and domain types.

Domains:

| Name    | Dataset                                                                        | Classes | Train / Test |
|---------|--------------------------------------------------------------------------------|---------|--------------|
| `birds` | [CUB-200](https://huggingface.co/datasets/cassiekang/cub200_dataset)           | 200     | 5,994 / 5,794 |
| `cars`  | [Stanford Cars](https://huggingface.co/datasets/tanganke/stanford_cars)        | 196     | 8,144 / 8,041 |

## Project layout

```
notebooks/                 exploration and scratch work
  00_baseline_sanity_check.ipynb
  01_lora_scratch.ipynb
  02_adapter_scratch.ipynb
src/llava_peft/            reusable code, importable as `llava_peft`
  utils.py                 .env + HF login, device selection (cuda > mps > cpu)
  data/                    dataset registry; every dataset loads as {image, label}
  model.py                 model/processor loading and generation
  methods/                 PEFT methods (lora.py, adapter.py)
  eval/metrics.py          scoring predictions against labels
scripts/                   command-line entry points (evaluate.py, later train.py)
results/                   run outputs (gitignored)
TODO.md                    roadmap
```

Load a dataset from anywhere:

```python
from llava_peft.data import get_dataset

spec = get_dataset("birds")
ds = spec.load("test", limit=100)   # columns: image, label
spec.class_names                    # all label strings
spec.prompt                         # question asked to the model
```

To add a domain, create `src/llava_peft/data/<name>.py` with a `DatasetSpec` and register it in `data/__init__.py`.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt   # also installs this repo in editable mode
```

Create a `.env` file in the project root with your Hugging Face token:

```
HF_TOKEN=hf_...
```

### Remote GPU (SSH)

Training runs on a remote GPU machine. On the remote:

```bash
git clone <repo-url> && cd Llava-Adapter
# same venv + pip install steps as above, then copy over .env
nvidia-smi                        # confirm the GPU is visible
python -c "from llava_peft.utils import get_device; print(get_device())"   # should print cuda
```

Run long jobs inside `tmux` so they survive a dropped connection. Set `HF_HOME` to a disk with space, since the 7B model is ~14GB. Copy `results/` back with `rsync` or `scp`.

## Running an evaluation

```bash
python scripts/evaluate.py --dataset birds --limit 200 --run-name birds_zeroshot
```

Outputs are written to `results/<run-name>/`.
