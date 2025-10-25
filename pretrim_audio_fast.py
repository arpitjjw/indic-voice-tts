#!/usr/bin/env python3
"""
FAST pre-trimming using parallel processing and optimized ffmpeg
"""

import json
import subprocess
import os
from pathlib import Path
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def trim_segment_batch(batch_data):
    """Process a batch of segments"""
    audio_file, segments_batch, output_dir, buffer_ms = batch_data
    results = {}

    for segment in segments_batch:
        # Calculate buffer boundaries
        buffer_start_ms = max(0, segment['start_ms'] - buffer_ms)
        buffer_end_ms = segment['end_ms'] + buffer_ms

        # Output file
        output_file = output_dir / f"segment_{segment['index']:04d}.mp3"

        # Use faster ffmpeg settings
        start_sec = buffer_start_ms / 1000
        duration_sec = (buffer_end_ms - buffer_start_ms) / 1000

        cmd = [
            'ffmpeg',
            '-ss', str(start_sec),  # Seek BEFORE input for faster processing
            '-i', audio_file,       # Input file AFTER seek
            '-t', str(duration_sec),
            '-c:a', 'libmp3lame',   # Use MP3 codec directly
            '-b:a', '64k',
            '-ar', '24000',
            '-ac', '1',
            '-threads', '1',        # Single thread per segment
            '-loglevel', 'panic',   # Minimal logging
            '-y',
            str(output_file)
        ]

        subprocess.run(cmd, capture_output=True)

        results[segment['index']] = {
            'file': str(output_file),
            'buffer_start_ms': buffer_start_ms,
            'buffer_end_ms': buffer_end_ms,
            'actual_start_ms': segment['start_ms'],
            'actual_end_ms': segment['end_ms']
        }

    return results

def main():
    # Paths
    json_path = "/home/arpit/speaker2_merged_transcripts_final.json"
    audio_path = "/home/arpit/Anuv Jain On TRS - Shaadi, Songs Aur Struggle Ki Kahani _ TRS [psRBjwgW7jA].mp3"
    output_dir = Path("/home/arpit/audio_segments")

    # Settings
    BUFFER_MS = 10000
    NUM_WORKERS = multiprocessing.cpu_count()  # Use all CPU cores

    # Create output directory
    if output_dir.exists():
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Using {NUM_WORKERS} parallel workers")

    # Load JSON data
    print(f"Loading segments from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data['segments']
    print(f"Found {len(segments)} segments to process")

    # First, convert MP3 to WAV for faster seeking (optional but can help)
    print("\n⚡ Converting to WAV for faster processing...")
    wav_path = output_dir / "temp_audio.wav"
    convert_cmd = [
        'ffmpeg', '-i', audio_path,
        '-ar', '24000',  # Resample once
        '-ac', '1',      # Convert to mono once
        '-loglevel', 'error',
        '-y',
        str(wav_path)
    ]
    subprocess.run(convert_cmd)
    print("✅ WAV conversion complete")

    # Split segments into batches for parallel processing
    batch_size = len(segments) // NUM_WORKERS + 1
    batches = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        batches.append((str(wav_path), batch, output_dir, BUFFER_MS))

    # Process in parallel
    print(f"\n🚀 Processing {len(segments)} segments in parallel...")
    segment_info = {}

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(trim_segment_batch, batch) for batch in batches]

        with tqdm(total=len(segments), desc="Trimming segments") as pbar:
            for future in as_completed(futures):
                results = future.result()
                segment_info.update(results)
                pbar.update(len(results))

    # Clean up temp WAV
    wav_path.unlink()

    # Save segment info
    info_file = output_dir / "segment_info.json"
    with open(info_file, 'w') as f:
        json.dump(segment_info, f, indent=2)

    print(f"\n✅ Complete! Trimmed {len(segment_info)} segments")
    print(f"📁 Audio files saved to: {output_dir}")
    print(f"📄 Segment info saved to: {info_file}")

    # Statistics
    total_size = sum(os.path.getsize(output_dir / f"segment_{i:04d}.mp3")
                    for i in segment_info.keys() if (output_dir / f"segment_{i:04d}.mp3").exists())
    print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")
    print(f"\n🚀 Now run: python audio_editor_fast.py")

if __name__ == "__main__":
    main()