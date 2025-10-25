"""
LoRA Finetuning script for CSM model adapted for existing pipeline.
Integrates with prepared_data from prepare_data.py and uses existing tokenization pipeline.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from safetensors.torch import save_file
import types
from typing import List, Dict, Any

# Import existing utilities
from utils import load_model, load_tokenizers, forward, WANDB_API_KEY
from dataloaders import TokenizedDataset, collate_fn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """LoRA adaptation for Linear layers"""

    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 32, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Original forward pass
        original_output = self.original_layer(x)

        # LoRA adaptation
        lora_output = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling

        return original_output + lora_output


def apply_lora_to_model(model, target_modules=None, rank=16, alpha=32, dropout=0.1):
    """Apply LoRA to specified modules in the CSM model"""

    if target_modules is None:
        # Default LoRA targets for CSM model based on csm-streaming
        target_modules = [
            "q_proj", "k_proj", "v_proj", "output_proj",  # Attention layers
            "w1", "w2", "w3",  # MLP layers
            "codebook0_head",  # Output heads
            "projection"
        ]

    lora_modules = {}

    def apply_lora_recursive(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this module should have LoRA applied
            should_apply = any(target in full_name for target in target_modules)

            if isinstance(child_module, nn.Linear) and should_apply:
                # Replace with LoRA version
                lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                lora_modules[full_name] = lora_layer
                logger.info(f"Applied LoRA to: {full_name}")
            else:
                # Recursively apply to children
                apply_lora_recursive(child_module, full_name)

    apply_lora_recursive(model)
    return lora_modules


def get_lora_parameters(model):
    """Get only the LoRA parameters for training"""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend(module.lora_A.parameters())
            lora_params.extend(module.lora_B.parameters())
    return lora_params


def print_trainable_parameters(model):
    """Print the number of trainable parameters"""
    trainable_params = 0
    all_param = 0
    lora_params = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()

    print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")
    print(f"LoRA params: {lora_params:,}")


def save_lora_weights(model, save_path):
    """Save only the LoRA weights"""
    lora_state_dict = {}
    lora_config = {'rank': None, 'alpha': None}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight
            lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight
            if lora_config['rank'] is None:
                lora_config['rank'] = module.rank
                lora_config['alpha'] = module.alpha

    torch.save({
        'lora_state_dict': lora_state_dict,
        'lora_config': lora_config
    }, save_path)
    logger.info(f"LoRA weights saved to: {save_path}")


def merge_lora_weights(model):
    """Merge LoRA weights into base model weights"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Merge: W = W + (alpha/r) * (B @ A)
            merged_delta = module.scaling * (module.lora_B.weight @ module.lora_A.weight)
            module.original_layer.weight.data += merged_delta

            # Zero out LoRA weights
            module.lora_A.weight.data.zero_()
            module.lora_B.weight.data.zero_()


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA finetuning for CSM model")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 tokenized data")
    parser.add_argument("--model_name_or_checkpoint_path", type=str, default=None, help="Model path or HuggingFace name")
    parser.add_argument("--output_dir", type=str, default="./lora_checkpoints", help="Output directory")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--load_in_memory", action="store_true", help="Load dataset in memory")
    parser.add_argument("--gen_every", type=int, default=500, help="Generate audio every N steps")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def validate_model(model, val_loader, device, use_amp=True):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            tokens, tokens_mask = batch
            tokens = tokens.to(device)
            tokens_mask = tokens_mask.to(device)

            with autocast(device_type=device, enabled=use_amp):
                loss = model(tokens, tokens_mask)

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_lora(args):
    """Main training function"""

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb if requested
    if args.use_wandb and WANDB_API_KEY:
        wandb.init(
            project="csm-lora-finetune",
            config=vars(args),
            name=f"lora_r{args.lora_rank}_alpha{args.lora_alpha}"
        )

    # Load model and tokenizers
    logger.info("Loading model and tokenizers...")
    device = torch.device(args.device)

    model = load_model(
        model_name_or_checkpoint_path=args.model_name_or_checkpoint_path,
        device=device
    )

    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    # Apply LoRA
    logger.info(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
    lora_modules = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )

    # Print parameter counts
    print_trainable_parameters(model)

    # Load datasets
    logger.info(f"Loading tokenized data from: {args.data}")
    train_dataset = TokenizedDataset(args.data, "train", load_in_memory=args.load_in_memory)
    val_dataset = TokenizedDataset(args.data, "val", load_in_memory=args.load_in_memory)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # Setup optimizer and scheduler
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * args.n_epochs // args.gradient_accumulation_steps

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Setup mixed precision
    scaler = GradScaler() if args.use_amp else None

    # Training loop
    logger.info("Starting LoRA training...")
    model.train()
    global_step = 0

    for epoch in range(args.n_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.n_epochs}")

        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            tokens, tokens_mask = batch
            tokens = tokens.to(device)
            tokens_mask = tokens_mask.to(device)

            with autocast(device_type=device, enabled=args.use_amp):
                loss = model(tokens, tokens_mask)
                loss = loss / args.gradient_accumulation_steps

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log to wandb
                if args.use_wandb and WANDB_API_KEY:
                    wandb.log({
                        "train_loss": loss.item() * args.gradient_accumulation_steps,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step
                    })

                progress_bar.set_postfix({
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

                # Generate audio sample periodically
                if global_step % args.gen_every == 0:
                    logger.info(f"Generating audio at step {global_step}")
                    try:
                        from utils import generate_audio
                        sample_text = "Hello, this is a test of the fine-tuned model."
                        sample_audio = generate_audio(
                            model, audio_tokenizer, text_tokenizer, None,
                            text=sample_text, speaker_id=2, device=device, use_amp=args.use_amp
                        )

                        # Save audio sample
                        import torchaudio
                        audio_path = os.path.join(args.output_dir, f"sample_step_{global_step}.wav")
                        torchaudio.save(audio_path, torch.tensor(sample_audio).unsqueeze(0), 24000)
                        logger.info(f"Audio sample saved to: {audio_path}")

                        if args.use_wandb and WANDB_API_KEY:
                            wandb.log({"audio_sample": wandb.Audio(audio_path, sample_rate=24000)})

                    except Exception as e:
                        logger.warning(f"Could not generate audio: {e}")

        # Validation
        val_loss = validate_model(model, val_loader, device, args.use_amp)
        logger.info(f"Epoch {epoch + 1} - Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if args.use_wandb and WANDB_API_KEY:
            wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"lora_checkpoint_epoch_{epoch + 1}.pt")
        save_lora_weights(model, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "lora_final.pt")
    save_lora_weights(model, final_path)

    # Also save merged model
    logger.info("Merging LoRA weights into base model...")
    merge_lora_weights(model)

    # Remove LoRA modules and save clean model
    def remove_lora_modules(module):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.original_layer)
            else:
                remove_lora_modules(child)

    remove_lora_modules(model)

    merged_path = os.path.join(args.output_dir, "merged_model.pt")
    torch.save({
        'model': model.state_dict(),
        'config': model.config if hasattr(model, 'config') else None
    }, merged_path)

    logger.info(f"Merged model saved: {merged_path}")

    if args.use_wandb and WANDB_API_KEY:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train_lora(args)