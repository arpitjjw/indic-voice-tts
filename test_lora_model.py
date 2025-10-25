"""
Test script for LoRA fine-tuned CSM model
"""

import torch
import argparse
import torchaudio
from pathlib import Path

from utils import load_model, load_tokenizers, generate_audio
from train_lora import LoRALinear, apply_lora_to_model


def load_lora_weights(model, lora_path):
    """Load LoRA weights into model"""
    checkpoint = torch.load(lora_path, map_location='cpu')
    lora_state_dict = checkpoint['lora_state_dict']
    lora_config = checkpoint['lora_config']

    print(f"Loading LoRA weights with rank={lora_config['rank']}, alpha={lora_config['alpha']}")

    for name, param in lora_state_dict.items():
        # Parse the name to find the correct module
        module_path = name.replace('.lora_A.weight', '').replace('.lora_B.weight', '')

        # Get the module
        module = model
        for attr in module_path.split('.'):
            module = getattr(module, attr)

        if '.lora_A.weight' in name:
            module.lora_A.weight.data.copy_(param)
        elif '.lora_B.weight' in name:
            module.lora_B.weight.data.copy_(param)

    print(f"LoRA weights loaded from: {lora_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test LoRA fine-tuned CSM model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--model_path", type=str, default="sesame/csm-1b", help="Base model path or HuggingFace model name")
    parser.add_argument("--text", type=str, default="Hello, this is a test of my fine-tuned voice.", help="Text to generate")
    parser.add_argument("--speaker_id", type=int, default=2, help="Speaker ID")
    parser.add_argument("--output_path", type=str, default="./lora_test_output.wav", help="Output audio path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base model
    print("Loading base model...")
    device = torch.device(args.device)
    model = load_model(args.model_path, device)

    # Load tokenizers
    print("Loading tokenizers...")
    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    # Apply LoRA structure (this creates the LoRA modules but with zero weights)
    print("Applying LoRA structure...")
    apply_lora_to_model(model, rank=16, alpha=32, dropout=0.1)

    # Load LoRA weights
    print("Loading LoRA weights...")
    load_lora_weights(model, args.lora_path)

    # Generate audio
    print(f"Generating audio for text: '{args.text}'")
    print(f"Speaker ID: {args.speaker_id}")

    try:
        audio = generate_audio(
            model=model,
            audio_tokenizer=audio_tokenizer,
            text_tokenizer=text_tokenizer,
            watermarker=None,  # You can load watermarker if needed
            text=args.text,
            speaker_id=args.speaker_id,
            device=device,
            use_amp=True
        )

        # Save audio
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            str(output_path),
            torch.tensor(audio).unsqueeze(0),
            24000
        )

        print(f"Audio generated and saved to: {output_path}")
        print(f"Audio duration: {len(audio) / 24000:.2f} seconds")

    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()