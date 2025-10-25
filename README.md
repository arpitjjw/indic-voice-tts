# CSM Voice Finetuning

Finetune Sesame AI's CSM-1B model for custom voices using **full finetuning** or **LoRA** (parameter-efficient).

## Features

- **Full Finetuning**: Modify original weights for significant domain shifts (new languages)
- **LoRA Finetuning**: Parameter-efficient training for voice adaptation
- **Efficient Training**: Pre-tokenization, compute amortization, padding-minimized batching
- **Hyperparameter Optimization**: Optuna sweeps with multi-GPU support
- **Data Utilities**: Audio preprocessing, dataset inspection tools

## Requirements

- Python 3.10
- CUDA-capable GPU
- `ffmpeg` (required for audio conversion in prepare_data.py)

## Installation

1. **Setup virtual environment:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Clone CSM repository:**
```bash
git clone https://github.com/SesameAILabs/csm.git ~/csm
cd ~/csm && git checkout 836f886515f0dec02c22ed2316cc78904bdc0f36
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your WANDB_API_KEY and CSM_REPO_PATH (e.g., ~/csm)
```

## Usage

### 1. Data Preparation

**Option A: From transcript JSON + audio file:**

First, create a transcript JSON with this format:
```json
{
  "speaker": "Speaker Name",
  "total_segments": 100,
  "segments": [
    {
      "text": "Your transcription here",
      "start_ms": 1000,
      "end_ms": 5200
    }
  ]
}
```

Then convert to training format:
```bash
python prepare_data.py \
  --transcript_json data.json \
  --audio_file audio.mp3 \
  --output_dir ./prepared_data \
  --speaker_id 3
```

This creates `train.json` and `val.json` in the output directory.

**Option B: Skip prepare_data.py if you already have the standard format:**

Create `train.json` and `val.json` directly with entries like:
```json
[
  {
    "text": "Your transcription",
    "path": "/path/to/audio.wav",
    "start": 1.0,
    "end": 5.2,
    "speaker": 3
  }
]
```

**Pre-tokenize for training:**
```bash
python pretokenize.py \
  --train_data prepared_data/train.json \
  --val_data prepared_data/val.json \
  --output data/tokens.hdf5
```

### 2. Training

**Full Finetuning:**
```bash
python train.py \
  --data data/tokens.hdf5 \
  --n_epochs 10 \
  --config configs/finetune_param_defaults.yaml \
  --gen_every 500
```

**LoRA Finetuning (recommended for single voice):**
```bash
python train_lora.py \
  --data data/tokens.hdf5 \
  --n_epochs 3 \
  --lora_rank 16 \
  --use_wandb
```

**Test LoRA Model:**
```bash
python test_lora_model.py \
  --lora_path lora_checkpoints/lora_final.pt \
  --text "Your test sentence here" \
  --speaker_id 3
```

### 3. Hyperparameter Optimization

```bash
python sweep.py \
  --data data/tokens.hdf5 \
  --sweep_config configs/sweep.yaml \
  --n_epochs 3 \
  --n_trials 20 \
  --n_gpus 1
```

## Key Parameters

| Parameter | Description | Default (train.py) | Default (train_lora.py) |
|-----------|-------------|---------|---------|
| `--batch_size` | Training batch size | 8 | 1 |
| `--learning_rate` | Learning rate | 4e-5 | 1e-4 |
| `--lora_rank` | LoRA rank (lower = fewer params) | N/A | 16 |
| `--use_amp` | Enable mixed precision | False | True |
| `--partial_data_loading` | For large datasets | False | N/A |
| `--load_in_memory` | Load full dataset in memory | False | False |
| `--omit_speaker_id` | Don't prepend speaker ID to text | False | N/A |

## Utilities

- `inspect_dataset.py` - View HDF5 dataset statistics
- `view_original_data.py` - View original JSON data before tokenization
- `pretrim_audio_fast.py` - Audio trimming utility

## Notes

- Audio files must be `.wav` format for pretokenize.py (use prepare_data.py to convert from mp3)
- Supported metadata formats for pretokenize.py: `.json`, `.csv`, `.parquet`, `.sql`
- Model supports up to 90 seconds of audio per segment
- Speaker IDs are prepended to text as `[speaker_id]` unless `--omit_speaker_id` is used