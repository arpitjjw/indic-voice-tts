"""
Preprocessing script to convert transcript JSON and audio to training format
"""

import json
import argparse
from pathlib import Path
import subprocess
import random
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for Sesame finetuning")
    parser.add_argument("--transcript_json", type=str, required=True, help="Path to speaker transcript JSON")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to source MP3/audio file")
    parser.add_argument("--output_dir", type=str, default="./prepared_data", help="Output directory for processed data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (0.1 = 10%)")
    parser.add_argument("--convert_to_wav", action="store_true", help="Convert MP3 to WAV format")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate for WAV conversion")
    parser.add_argument("--min_duration", type=float, default=2.0, help="Minimum segment duration in seconds")
    parser.add_argument("--max_duration", type=float, default=89.0, help="Maximum segment duration in seconds (model limit is 90)")
    parser.add_argument("--speaker_id", type=int, default=None, help="Override speaker ID for all segments (e.g., 3)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for train/val split")
    return parser.parse_args()

def convert_mp3_to_wav(mp3_path, wav_path, sample_rate=24000):
    """Convert MP3 to WAV using ffmpeg"""
    print(f"Converting {mp3_path} to WAV format at {sample_rate}Hz...")
    cmd = [
        "ffmpeg", "-i", mp3_path,
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
        "-y",  # overwrite
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error converting audio: {result.stderr}")
        raise Exception("Failed to convert audio file")
    print(f"Audio converted successfully to: {wav_path}")
    return wav_path

def load_transcript_json(json_path):
    """Load the transcript JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_to_training_format(transcript_data, audio_path, min_duration=2.0, max_duration=89.0, override_speaker_id=None):
    """Convert transcript format to training format

    Args:
        transcript_data: Original transcript data
        audio_path: Path to audio file
        min_duration: Minimum segment duration in seconds (default 2.0)
        max_duration: Maximum segment duration in seconds (default 89.0, model limit is 90)
        override_speaker_id: Override speaker ID for all segments (optional)
    """
    segments = []
    skipped_too_short = 0
    skipped_too_long = 0

    # If override_speaker_id is provided, use it for all segments
    if override_speaker_id is not None:
        speaker_num = override_speaker_id
        print(f"  Using override speaker ID: {speaker_num}")
    else:
        speaker_id = transcript_data.get("speaker", "Speaker 2")
        # Extract speaker number if format is "Speaker N"
        if "Speaker" in speaker_id:
            try:
                speaker_num = int(speaker_id.split()[-1])
            except:
                speaker_num = 2  # default
        else:
            speaker_num = 2
        print(f"  Using speaker ID from data: {speaker_num}")

    for segment in transcript_data["segments"]:
        duration = (segment["end_ms"] - segment["start_ms"]) / 1000.0

        # Skip segments that are too short or too long
        if duration < min_duration:
            skipped_too_short += 1
            continue
        if duration > max_duration:
            skipped_too_long += 1
            continue

        converted_segment = {
            "text": segment["text"],
            "path": str(audio_path),
            "start": segment["start_ms"] / 1000.0,  # Convert ms to seconds
            "end": segment["end_ms"] / 1000.0,
            "speaker": speaker_num
        }
        segments.append(converted_segment)

    print(f"  Filtered segments: {len(segments)} kept, {skipped_too_short} too short (<{min_duration}s), {skipped_too_long} too long (>{max_duration}s)")
    return segments

def split_train_val(segments, val_split=0.1, random_seed=42):
    """Split segments into training and validation sets"""
    random.seed(random_seed)

    # Shuffle segments
    shuffled_segments = segments.copy()
    random.shuffle(shuffled_segments)

    # Calculate split
    val_size = int(len(shuffled_segments) * val_split)
    val_segments = shuffled_segments[:val_size]
    train_segments = shuffled_segments[val_size:]

    return train_segments, val_segments

def save_json(data, output_path):
    """Save data as JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} segments to: {output_path}")

def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load transcript
    print(f"Loading transcript from: {args.transcript_json}")
    transcript_data = load_transcript_json(args.transcript_json)
    print(f"Found {transcript_data['total_segments']} segments for {transcript_data['speaker']}")

    # Handle audio file
    audio_path = args.audio_file
    if args.convert_to_wav or audio_path.endswith('.mp3'):
        # Convert to WAV
        wav_filename = Path(audio_path).stem + ".wav"
        wav_path = output_dir / wav_filename
        audio_path = convert_mp3_to_wav(audio_path, str(wav_path), args.sample_rate)

    # Convert to training format
    print("\nConverting to training format...")
    segments = convert_to_training_format(transcript_data, audio_path, args.min_duration, args.max_duration, args.speaker_id)
    print(f"Total segments after filtering: {len(segments)}")

    # Split into train/val
    print(f"\nSplitting data (validation: {args.val_split*100:.0f}%)...")
    train_segments, val_segments = split_train_val(segments, args.val_split, args.random_seed)
    print(f"Train: {len(train_segments)} segments")
    print(f"Validation: {len(val_segments)} segments")

    # Save JSON files
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    save_json(train_segments, train_path)
    save_json(val_segments, val_path)

    # Print summary
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"Audio file: {audio_path}")
    print(f"Train data: {train_path}")
    print(f"Validation data: {val_path}")

    # Show example of first segment
    print("\nExample segment (first training sample):")
    if train_segments:
        example = train_segments[0]
        print(f"  Text: {example['text'][:100]}...")
        print(f"  Audio: {example['path']}")
        print(f"  Time: {example['start']:.2f}s - {example['end']:.2f}s")
        print(f"  Speaker: {example['speaker']}")

    # Print next steps
    print("\n" + "="*50)
    print("Next steps:")
    print("="*50)
    print("1. Pre-tokenize the data:")
    print(f"   python pretokenize.py --train_data {train_path} --val_data {val_path} --output {output_dir}/tokens.hdf5")
    print("\n2. Fine-tune the model:")
    print(f"   python train.py --data {output_dir}/tokens.hdf5 --n_epochs 10")
    print(f"   # Or with custom checkpoint: --model_name_or_checkpoint_path /path/to/model.pt")
    print("\nNote: Adjust paths and parameters as needed for your setup.")

if __name__ == "__main__":
    main()