# CSM Voice Finetuning

Finetune Sesame AI's CSM-1B model for custom voices using **full finetuning** or **LoRA** (parameter-efficient).

Based on [knottwill/sesame-finetune](https://github.com/knottwill/sesame-finetune) with added LoRA support and data preparation utilities.

## Features

- **Full Finetuning**: Modify original weights for significant domain shifts (new languages)
- **LoRA Finetuning**: Parameter-efficient training for voice adaptation
- **Efficient Training**: Pre-tokenization, compute amortization, padding-minimized batching
- **Hyperparameter Optimization**: Optuna sweeps with multi-GPU support
- **Data Utilities**: Audio preprocessing, dataset inspection tools

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

**From transcript JSON + audio file:**
```bash
python prepare_data.py \
  --transcript_json data.json \
  --audio_file audio.mp3 \
  --output_dir ./prepared_data \
  --speaker_id 3
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

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--batch_size` | Training batch size | 8 |
| `--learning_rate` | Learning rate | 4e-5 |
| `--lora_rank` | LoRA rank (lower = fewer params) | 16 |
| `--use_amp` | Enable mixed precision | False |
| `--partial_data_loading` | For large datasets | False |
| `--omit_speaker_id` | Don't prepend speaker ID to text | False |

## Utilities

- `inspect_dataset.py` - View HDF5 dataset statistics
- `view_original_data.py` - View original JSON data before tokenization
- `pretrim_audio_fast.py` - Audio trimming utility

## Data Format

Your metadata JSON should contain entries with:
```json
{
  "text": "Transcription text here",
  "path": "/path/to/audio.wav",
  "start": 0.0,
  "end": 5.2,
  "speaker": 3
}
```

Supported formats: `.json`, `.csv`, `.parquet`, `.sql`