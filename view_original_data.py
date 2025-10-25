"""
Script to view the original JSON data that was used to create the HDF5 dataset
Shows the actual text and metadata before tokenization
"""

import json
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="View original training data")
    parser.add_argument("--train_json", type=str, default="./prepared_data/train.json", help="Path to train JSON")
    parser.add_argument("--val_json", type=str, default="./prepared_data/val.json", help="Path to validation JSON")
    parser.add_argument("--show_samples", type=int, default=10, help="Number of samples to show")
    parser.add_argument("--export_csv", type=str, help="Export to CSV file for easier viewing")
    return parser.parse_args()

def view_json_data(json_path, split_name, show_samples=10):
    """View JSON training data"""

    print(f"\n{'='*60}")
    print(f"{split_name.upper()} DATA: {json_path}")
    print('='*60)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Calculate statistics
    durations = [(item['end'] - item['start']) for item in data]
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    total_duration = sum(durations)

    print(f"Average duration: {avg_duration:.2f} seconds")
    print(f"Min duration: {min_duration:.2f} seconds")
    print(f"Max duration: {max_duration:.2f} seconds")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")

    # Get unique speakers
    speakers = set(item.get('speaker', 'unknown') for item in data)
    print(f"Unique speakers: {speakers}")

    # Show samples
    print(f"\nFirst {min(show_samples, len(data))} samples:")
    print("-"*60)

    for i, item in enumerate(data[:show_samples]):
        duration = item['end'] - item['start']
        text_preview = item['text'][:100] + "..." if len(item['text']) > 100 else item['text']

        print(f"\nSample {i+1}:")
        print(f"  Speaker: {item.get('speaker', 'unknown')}")
        print(f"  Duration: {duration:.2f}s (from {item['start']:.2f}s to {item['end']:.2f}s)")
        print(f"  Text length: {len(item['text'])} chars")
        print(f"  Text: {text_preview}")
        print(f"  Audio: {Path(item['path']).name}")

    return data

def export_to_csv(train_data, val_data, csv_path):
    """Export train and val data to CSV for easier viewing"""

    # Prepare data for DataFrame
    train_df_data = []
    for item in train_data:
        train_df_data.append({
            'split': 'train',
            'speaker': item.get('speaker', 'unknown'),
            'start_time': item['start'],
            'end_time': item['end'],
            'duration': item['end'] - item['start'],
            'text': item['text'],
            'audio_file': Path(item['path']).name
        })

    val_df_data = []
    for item in val_data:
        val_df_data.append({
            'split': 'val',
            'speaker': item.get('speaker', 'unknown'),
            'start_time': item['start'],
            'end_time': item['end'],
            'duration': item['end'] - item['start'],
            'text': item['text'],
            'audio_file': Path(item['path']).name
        })

    # Create DataFrame
    df = pd.DataFrame(train_df_data + val_df_data)

    # Save to CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n✅ Data exported to: {csv_path}")
    print(f"   Total rows: {len(df)}")
    print(f"   Train samples: {len(train_df_data)}")
    print(f"   Val samples: {len(val_df_data)}")

if __name__ == "__main__":
    args = parse_args()

    # View train data
    train_data = view_json_data(args.train_json, "train", args.show_samples)

    # View validation data
    val_data = view_json_data(args.val_json, "validation", args.show_samples)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total samples: {len(train_data) + len(val_data)}")
    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Train/Val ratio: {len(train_data)/len(val_data):.2f}:1")

    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(train_data, val_data, args.export_csv)