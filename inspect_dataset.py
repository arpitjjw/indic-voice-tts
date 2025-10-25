"""
Script to inspect the HDF5 tokenized dataset
Shows statistics and sample data from train/val splits
"""

import h5py
import numpy as np
import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect HDF5 tokenized dataset")
    parser.add_argument("--data", type=str, default="./prepared_data/tokens.hdf5", help="Path to HDF5 file")
    parser.add_argument("--show_samples", type=int, default=5, help="Number of samples to show")
    parser.add_argument("--export_json", type=str, help="Export original text to JSON file")
    return parser.parse_args()

def decode_speaker_from_text(text_tokens):
    """Try to extract speaker ID from text tokens (if present in format [speaker_id])"""
    # This is a simplified version - actual decoding would need the tokenizer
    # But we can still see the pattern in the tokens
    return None  # Placeholder

def inspect_hdf5(file_path, show_samples=5, export_json=None):
    """Inspect the HDF5 dataset"""

    print(f"Inspecting HDF5 file: {file_path}")
    print("=" * 60)

    with h5py.File(file_path, "r") as f:
        # Show available splits
        splits = list(f.keys())
        print(f"Available splits: {splits}\n")

        all_data = {}

        for split in splits:
            print(f"\n{split.upper()} Split:")
            print("-" * 40)

            # Get datasets
            audio = f[f"{split}/audio"]
            text = f[f"{split}/text"]
            length = f[f"{split}/length"]

            num_samples = len(audio)
            print(f"Number of samples: {num_samples}")

            # Calculate statistics
            lengths = np.array(length[:])
            print(f"Average sequence length: {np.mean(lengths):.2f}")
            print(f"Min sequence length: {np.min(lengths)}")
            print(f"Max sequence length: {np.max(lengths)}")
            print(f"Total tokens: {np.sum(lengths)}")

            # Show duration estimates (approximate)
            # Assuming ~50 audio tokens per second (this is approximate)
            audio_lengths = []
            for i in range(min(100, num_samples)):  # Sample first 100 for speed
                audio_tokens = audio[i]
                # Audio tokens are flattened, so divide by num_codebooks (32) to get sequence length
                seq_len = len(audio_tokens) // 32
                audio_lengths.append(seq_len)

            avg_audio_tokens = np.mean(audio_lengths)
            estimated_duration_seconds = avg_audio_tokens / 50  # Rough estimate
            print(f"Average audio tokens per sample: {avg_audio_tokens:.2f}")
            print(f"Estimated average duration: {estimated_duration_seconds:.2f} seconds")

            # Show sample data
            if show_samples > 0:
                print(f"\nFirst {min(show_samples, num_samples)} samples:")
                print("-" * 40)

                samples_to_export = []
                for i in range(min(show_samples, num_samples)):
                    audio_tokens = audio[i]
                    text_tokens = text[i]
                    total_length = length[i]

                    # Calculate audio duration
                    audio_seq_len = len(audio_tokens) // 32

                    print(f"\nSample {i+1}:")
                    print(f"  Total length: {total_length}")
                    print(f"  Text tokens: {len(text_tokens)}")
                    print(f"  Audio tokens (flattened): {len(audio_tokens)}")
                    print(f"  Audio sequence length: {audio_seq_len}")
                    print(f"  Estimated duration: {audio_seq_len/50:.2f}s")
                    print(f"  Text token preview: {text_tokens[:20]}..." if len(text_tokens) > 20 else f"  Text tokens: {text_tokens}")

                    # Store for export
                    samples_to_export.append({
                        "index": i,
                        "text_token_length": len(text_tokens),
                        "audio_token_length": audio_seq_len,
                        "estimated_duration_seconds": audio_seq_len/50,
                        "text_tokens_preview": text_tokens[:50].tolist() if len(text_tokens) > 50 else text_tokens.tolist()
                    })

                all_data[split] = {
                    "num_samples": num_samples,
                    "avg_length": float(np.mean(lengths)),
                    "samples": samples_to_export
                }

    # Export to JSON if requested
    if export_json:
        with open(export_json, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"\nData exported to: {export_json}")

    print("\n" + "=" * 60)
    print("Inspection complete!")

    # Summary
    total_samples = sum(len(f[f"{split}/audio"]) for split in splits)
    print(f"\nTotal samples across all splits: {total_samples}")

if __name__ == "__main__":
    args = parse_args()
    inspect_hdf5(args.data, args.show_samples, args.export_json)