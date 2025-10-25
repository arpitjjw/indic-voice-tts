#!/bin/bash

# LoRA Fine-tuning Script for CSM Model
# Usage: ./run_lora_training.sh

set -e

echo "Starting LoRA fine-tuning for CSM model..."

# Configuration
DATA_PATH="./prepared_data/tokens.hdf5"
MODEL_PATH="/home/arpit/finetune/model/model_8400.pt"
OUTPUT_DIR="./lora_checkpoints"
EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
LEARNING_RATE=1e-4
LORA_RANK=16
LORA_ALPHA=32

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Tokenized data not found at $DATA_PATH"
    echo "Please run pretokenization first:"
    echo "python pretokenize_parallel.py --train_data ./prepared_data/train.json --val_data ./prepared_data/val.json --output $DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Training Configuration:"
echo "  Data: $DATA_PATH"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  LoRA Rank: $LORA_RANK"
echo "  LoRA Alpha: $LORA_ALPHA"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Run LoRA training
python train_lora.py \
    --data "$DATA_PATH" \
    --model_name_or_checkpoint_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.1 \
    --warmup_steps 100 \
    --max_grad_norm 1.0 \
    --gen_every 500 \
    --use_amp \
    --load_in_memory \
    --use_wandb

echo ""
echo "LoRA fine-tuning completed!"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo ""
echo "To test the fine-tuned model, run:"
echo "python test_lora_model.py --lora_path $OUTPUT_DIR/lora_final.pt"