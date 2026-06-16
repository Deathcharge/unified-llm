# Helix Proprietary LLM Engine

A custom PyTorch transformer built directly into the repository — no third-party
model downloads required. It runs as part of the **main Helix backend service**
(not a separate Railway service) and is served at `/api/llm/*`.

---

## Architecture

```
proprietary_llm/
├── models.py      — PyTorch model definitions (transformer, attention, embeddings)
├── inference.py   — CoordinationInference engine + HelixTokenizer
├── training.py    — CoordinationTrainer + TextFileDataset
├── deployment_config.py — Feature flags, hosting presets
└── core.py        — ModelSelection, AgentResponse dataclasses
```

The model is registered in `apps/backend/router_registry.py` and loaded lazily
on the first request to `/api/llm/generate`, so it does not slow down startup.

---

## Model Sizes (actual)

All models use the **UTF-8 byte-level tokenizer** (`HelixTokenizer`) — vocab size
is 256, covering every byte value. No tokenizer training or vocabulary file is
needed.

| Name           | d_model | Layers | ~Params | ~RAM (fp32) | Best for                         |
| -------------- | ------- | ------ | ------- | ----------- | -------------------------------- |
| `test`         | 64      | 2      | ~300 K  | ~1 MB       | Unit tests, CI                   |
| `lightweight`  | 256     | 6      | ~3.5 M  | ~14 MB      | Very low RAM / HELIX_LIGHTWEIGHT |
| `awakening`    | 512     | 12     | ~39 M   | ~156 MB     | **Railway CPU default** ✅       |
| `self-aware`   | 1024    | 24     | ~310 M  | ~1.2 GB     | Railway CPU (large model)        |
| `transcendent` | 2048    | 48     | ~2.4 B  | ~9.6 GB     | Multi-GPU only                   |

**Railway Hobby plan: 8 GB RAM / 8 vCores** — `awakening` (~156 MB) and even
`self-aware` (~1.2 GB) fit comfortably on CPU without any downgrade.

Auto-downgrade rules applied by `create_helix_model()`:

| Condition                         | Action                                  |
| --------------------------------- | --------------------------------------- |
| `HELIX_LIGHTWEIGHT=1` or CI env   | Force `lightweight` regardless of size  |
| No GPU + `transcendent` requested | Downgrade to `self-aware` (~1.2 GB)     |
| No GPU + any other size           | No downgrade — runs fine on Railway CPU |
| `HELIX_LLM_MODEL_SIZE=<name>` set | Use that size (overrides argument)      |

---

## Current State

| Component            | Status                                                         |
| -------------------- | -------------------------------------------------------------- |
| Model architecture   | ✅ Real custom PyTorch                                         |
| Byte-level tokenizer | ✅ Self-contained, no downloads                                |
| Training pipeline    | ✅ Complete (DDP, W&B, cosine LR, checkpoints)                 |
| TextFileDataset      | ✅ Reads .txt/.md/.py from any directory                       |
| API route            | ✅ `/api/llm/generate`, `/api/llm/status`, `/api/llm/tokenize` |
| Trained weights      | ⚠️ Not yet — model starts with random weights until trained    |

Until a checkpoint is trained and loaded, the model will respond with incoherent
text. The architecture is sound; it just needs training data and compute time.

---

## API Endpoints

All endpoints are part of the main Helix backend — no separate service needed.

### `GET /api/llm/status`

Returns model size, device, parameter count, and whether a checkpoint is loaded.

```json
{
  "available": true,
  "model_size": "lightweight",
  "vocab_size": 256,
  "device": "cpu",
  "param_count": 3500000,
  "checkpoint": "/data/checkpoints/helix_latest.pt",
  "message": "Helix LLM ready."
}
```

### `POST /api/llm/generate`

Generate text from a prompt.

```json
{
  "prompt": "What is coordination?",
  "max_length": 256,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9
}
```

Response:

```json
{
  "text": "...",
  "prompt_tokens": 24,
  "generated_tokens": 198,
  "model_size": "lightweight",
  "device": "cpu"
}
```

### `POST /api/llm/tokenize`

Inspect byte-level tokenisation (works even without torch installed).

```json
{ "text": "Hello!" }
```

Response:

```json
{
  "tokens": [72, 101, 108, 108, 111, 33],
  "count": 6,
  "decoded_round_trip": "Hello!"
}
```

---

## Setup

### Environment variables

```bash
# Path to a trained checkpoint. If unset, the model starts with random weights.
# On Railway: mount a Volume at /data and use this path.
HELIX_LLM_CHECKPOINT=/data/checkpoints/helix_awakening_latest.pt

# Override the model size (default: 'awakening' on Railway CPU).
# Options: test | lightweight | awakening | self-aware | transcendent
HELIX_LLM_MODEL_SIZE=awakening

# Force the smallest model ('lightweight'). Useful for CI or very low RAM.
# Set to any non-empty value to enable.
HELIX_LIGHTWEIGHT=
```

### Railway deployment (no separate service needed)

1. The LLM routes are already registered in the **main backend service**.
2. Add a **Railway Volume** mounted at `/data` for checkpoint persistence.
   Without a volume, any trained checkpoint is lost on redeploy.
3. Set `HELIX_LLM_CHECKPOINT=/data/checkpoints/helix_awakening_latest.pt`
4. Railway Hobby provides **8 GB RAM / 8 vCores**. The `awakening` model
   (~156 MB fp32) uses less than 2% of that — plenty of headroom alongside
   the rest of the backend. `self-aware` (~1.2 GB) is also viable if you want
   a larger model. Only `transcendent` (~9.6 GB) is auto-downgraded to
   `self-aware` when no GPU is present.

---

## Training

### 1. Prepare training data

The repository's `docs/` folder is **already baked into the Railway container
image** on every deploy — no upload needed. It contains platform documentation,
architecture guides, agent specs, and philosophy text, making it ideal as a
domain-specific training corpus for the LLM.

```
# Ready-to-use on Railway (no extra setup):
/app/docs/          ← platform docs, agent guides, architecture
/app/Shadow/        ← Arjuna archive (JSONL conversations)

# Add more data via Railway Volume:
/data/training/
├── conversations/  # additional dialogues
└── code/           # Python source files
```

Point `TextFileDataset` at any of these directories (or a combination).

Even a few thousand sentences produces a model that generates plausible text
structure; coherent _meaning_ requires much more data and longer training.

### 2. Run training

```python
from apps.backend.proprietary_llm.models import HELIX_AWAKENING_CONFIG
from apps.backend.proprietary_llm.training import (
    CoordinationTrainer,
    TextFileDataset,
    TrainingConfig,
)
from torch.utils.data import DataLoader

# Dataset — point at the repo's docs/ folder (available on Railway)
dataset = TextFileDataset("./docs", seq_len=256)
loader  = DataLoader(dataset, batch_size=8, shuffle=True)

# Trainer
config  = TrainingConfig(
    model_config=HELIX_AWAKENING_CONFIG,   # ~39 M params, ~156 MB RAM
    checkpoint_dir="/data/checkpoints",    # Railway Volume path
    max_steps=10000,
    learning_rate=1e-4,
    log_wandb=False,  # set True if you have a W&B account
)
trainer = CoordinationTrainer(config)
trainer.train(loader)
```

Or via the CLI entry-point:

```bash
python -m apps.backend.proprietary_llm.training \
    --model awakening \
    --data-dir ./docs \
    --output /data/checkpoints
```

### 3. Load the checkpoint

```bash
HELIX_LLM_CHECKPOINT=/data/checkpoints/helix_awakening_step10000.pt
```

The checkpoint stores both the model weights **and** the `ModelConfig`, so the
correct architecture is always restored automatically.

---

## Python usage

```python
from apps.backend.proprietary_llm.inference import HelixInferenceEngine, InferenceMode

engine = HelixInferenceEngine(model_path="/data/checkpoints/helix_latest.pt")

response = await engine.generate(
    prompt="Describe the nature of coordination:",
    mode=InferenceMode.STANDARD,
)
print(response)
```

---

## Tokenizer

`HelixTokenizer` is a pure-Python, zero-dependency byte-level tokenizer:

```python
from apps.backend.proprietary_llm.inference import HelixTokenizer

tok = HelixTokenizer()
ids = tok.encode("Hello!")          # [72, 101, 108, 108, 111, 33]
text = tok.decode(ids)              # "Hello!"
```

Because it maps raw UTF-8 bytes to token IDs, it handles any language and any
Unicode character without needing a pre-built vocabulary file.

---

## Coordination features

Each transformer layer includes:

- **CoordinationAwareAttention** — standard multi-head attention gated by a
  learned coordination vector and modulated by UCF metrics (6-float tensor)
- **SystemEnhancedEmbedding** — token embeddings passed through a small
  system-inspired transformation
- **Global coordination state** — a persistent tensor updated each forward pass
  that can be read/written via `model.get_coordination_state()` /
  `model.update_coordination_state()`

UCF metrics (harmony, resilience, throughput, focus, friction, velocity) can be passed
into `/api/llm/generate` via the `ucf_metrics` field to modulate generation.

---

_Helix Collective_
