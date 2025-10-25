"""
Parallel GPU-optimized pretokenization script for Sesame finetuning.
Optimized for A100 GPU with batch processing and multiprocessing.

Usage:
python pretokenize_parallel.py --train_data ./sesame_hindi_final/train.json --val_data ./sesame_hindi_final/val.json --output ./hindi_tokens.hdf5
"""

import argparse
from pathlib import Path
import sqlite3
import pandas as pd
import torch
import torchaudio
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from typing import List, Dict, Tuple

from utils import load_tokenizers, MIMI_SAMPLE_RATE, AUDIO_NUM_CODEBOOKS


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="./data/tokens.hdf5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for GPU processing (recommended: 32-64 for A100)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers for data loading")
    parser.add_argument("--save_every", type=int, default=500, help="Save every N batches")
    parser.add_argument("--omit_speaker_id", action="store_true", help="Don't prepend text with a speaker id")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Use pinned memory for faster GPU transfer")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor")
    
    args = parser.parse_args(arg_string.split() if arg_string else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


class AudioDataset(Dataset):
    """Dataset for loading audio files and metadata for parallel processing."""
    
    def __init__(self, metadata_df: pd.DataFrame, omit_speaker_id: bool = False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.omit_speaker_id = omit_speaker_id
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        
        # Handle optional timestamps
        if "start" in row and "end" in row and pd.notna(row["start"]) and pd.notna(row["end"]):
            audio_info = torchaudio.info(row["path"])
            frame_offset = int(row["start"] * audio_info.sample_rate)
            num_frames = int((row["end"] - row["start"]) * audio_info.sample_rate)
        else:
            frame_offset = 0
            num_frames = -1
        
        return {
            "path": row["path"],
            "text": row["text"],
            "speaker": row.get("speaker", 999),
            "frame_offset": frame_offset,
            "num_frames": num_frames,
            "index": idx
        }


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, List[str], List[int], List[int]]:
    """
    Collate function to batch audio loading and create text tokens.
    Returns: (batched_waveforms, text_list, speaker_list, indices)
    """
    waveforms = []
    texts = []
    speakers = []
    indices = []
    
    for item in batch:
        # Load and resample audio
        try:
            waveform, sr = torchaudio.load(
                item["path"], 
                frame_offset=item["frame_offset"], 
                num_frames=item["num_frames"]
            )
            # Resample to Mimi sample rate
            if sr != MIMI_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform.squeeze(0), 
                    orig_freq=sr, 
                    new_freq=MIMI_SAMPLE_RATE
                )
            else:
                waveform = waveform.squeeze(0)
            
            # Ensure mono and correct shape for Mimi: [1, seq_len]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [seq_len] -> [1, seq_len]
            elif waveform.dim() == 2:
                waveform = waveform.mean(dim=0, keepdim=True)  # [channels, seq_len] -> [1, seq_len]
            
            waveforms.append(waveform)
            texts.append(item["text"])
            speakers.append(item["speaker"])
            indices.append(item["index"])
            
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            continue
    
    if not waveforms:
        return None
    
    # Pad waveforms to same length for batch processing
    max_length = max(w.shape[-1] for w in waveforms)
    padded_waveforms = []
    for w in waveforms:
        if w.shape[-1] < max_length:
            padding = max_length - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, padding))
        padded_waveforms.append(w)
    
    # Stack into batch: [batch_size, 1, seq_len] - Each waveform is [1, seq_len]
    batched_waveforms = torch.stack(padded_waveforms, dim=0)  # [batch_size, 1, seq_len]
    
    return batched_waveforms, texts, speakers, indices


def load_metadata(data_path: Path | str) -> pd.DataFrame:
    """Load metadata from various formats."""
    if isinstance(data_path, str):
        data_path = Path(data_path)

    if data_path.suffix == ".json":
        return pd.read_json(data_path)
    elif data_path.suffix == ".csv":
        return pd.read_csv(data_path)
    elif data_path.suffix == ".sql":
        return pd.read_sql_query("SELECT * FROM data", sqlite3.connect(data_path))
    elif data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    elif data_path.suffix == ".pkl":
        return pd.read_pickle(data_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {data_path}")


def append_to_hdf5(file_path, split, audio_tokens_batch, text_tokens_batch, compression="gzip"):
    """Append audio, text, and length information to the HDF5 file."""
    with h5py.File(file_path, "a") as f:
        grp = f.require_group(split)

        vlen_dtype = h5py.special_dtype(vlen=np.int32)
        audio_ds = grp.get("audio") or grp.create_dataset("audio", shape=(0,), maxshape=(None,), dtype=vlen_dtype)
        text_ds = grp.get("text") or grp.create_dataset("text", shape=(0,), maxshape=(None,), dtype=vlen_dtype)
        length_ds = grp.get("length") or grp.create_dataset("length", shape=(0,), maxshape=(None,), dtype=np.int32)

        n = len(audio_tokens_batch)
        audio_ds.resize(audio_ds.shape[0] + n, axis=0)
        text_ds.resize(text_ds.shape[0] + n, axis=0)
        length_ds.resize(length_ds.shape[0] + n, axis=0)

        for i in range(n):
            audio_array = np.array(audio_tokens_batch[i], dtype=np.int32).flatten()
            text_array = np.array(text_tokens_batch[i], dtype=np.int32)

            seq_len = audio_array.shape[0] // AUDIO_NUM_CODEBOOKS
            total_len = seq_len + len(text_array) + 1  # +1 for EOS frame

            audio_ds[-n + i] = audio_array
            text_ds[-n + i] = text_array
            length_ds[-n + i] = total_len


def get_num_existing_samples(file_path, split):
    """Return the number of existing samples in the HDF5 file for the given split."""
    try:
        with h5py.File(file_path, "r") as f:
            return f[split]["length"].shape[0]
    except Exception:
        return 0


def tokenize_and_store_parallel(
    data_path, output_path, split, audio_tokenizer, text_tokenizer, 
    device, batch_size=32, num_workers=8, save_every=500, 
    omit_speaker_id=False, pin_memory=True, prefetch_factor=4
):
    """
    Parallel tokenization with GPU batch processing.
    Optimized for A100 GPU with efficient DataLoader.
    """
    df = load_metadata(data_path)
    n_existing = get_num_existing_samples(output_path, split)
    
    if n_existing:
        print(f"⏩ Resuming {split}: skipping {n_existing} already processed samples")
        df = df.iloc[n_existing:]
    else:
        print(f"🔄 Processing {split} split: {len(df)} samples")

    if len(df) == 0:
        print(f"✅ {split} split already complete")
        return

    # Create dataset and dataloader
    dataset = AudioDataset(df, omit_speaker_id=omit_speaker_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: maintain order for resuming
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=False
    )

    audio_tokens_batch = []
    text_tokens_batch = []
    batch_count = 0
    
    print(f"🚀 Starting parallel processing with {num_workers} workers, batch_size={batch_size}")
    
    with tqdm(total=len(dataloader), desc=f"Processing {split}") as pbar:
        for batch_data in dataloader:
            if batch_data is None:  # Skip failed batches
                pbar.update(1)
                continue
                
            batched_waveforms, texts, speakers, indices = batch_data
            batched_waveforms = batched_waveforms.to(device, non_blocking=True)
            
            # Batch audio tokenization on GPU
            with torch.no_grad():
                # Mimi encode expects [batch_size, 1, seq_len]
                audio_tokens_list = audio_tokenizer.encode(batched_waveforms)
                
                # Convert to list format for each sample
                for i, audio_tokens in enumerate(audio_tokens_list):
                    # audio_tokens shape: [n_codebooks, seq_len]
                    audio_tokens_batch.append(audio_tokens.cpu().tolist())
                    
                    # Process text tokens
                    speaker = speakers[i]
                    text = f"[{speaker}]{texts[i]}" if not omit_speaker_id else texts[i]
                    text_tokens = text_tokenizer.encode(text)
                    text_tokens_batch.append(text_tokens)

            batch_count += 1
            
            # Save periodically
            if batch_count % save_every == 0:
                print(f"\n💾 Saving checkpoint at batch {batch_count}")
                append_to_hdf5(output_path, split, audio_tokens_batch, text_tokens_batch)
                audio_tokens_batch, text_tokens_batch = [], []
            
            pbar.update(1)
            pbar.set_postfix({
                'GPU_Batch': batch_size,
                'Workers': num_workers,
                'Saved': len(audio_tokens_batch)
            })

    # Final flush
    if audio_tokens_batch:
        print(f"\n💾 Final save: {len(audio_tokens_batch)} samples")
        append_to_hdf5(output_path, split, audio_tokens_batch, text_tokens_batch)


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"🎯 Device: {device}")
    print(f"📊 Batch size: {args.batch_size} (recommended: 64 for A100 fp16)")
    print(f"👥 Workers: {args.num_workers}")
    print(f"📌 Pin memory: {args.pin_memory}")
    
    # Load tokenizers
    print("🔧 Loading tokenizers...")
    text_tokenizer, audio_tokenizer = load_tokenizers(device)
    
    # Process training data
    tokenize_and_store_parallel(
        args.train_data, output_path=args.output, split="train",
        audio_tokenizer=audio_tokenizer, text_tokenizer=text_tokenizer,
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        save_every=args.save_every, omit_speaker_id=args.omit_speaker_id,
        pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor
    )

    # Process validation data
    tokenize_and_store_parallel(
        args.val_data, output_path=args.output, split="val",
        audio_tokenizer=audio_tokenizer, text_tokenizer=text_tokenizer,
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        save_every=args.save_every, omit_speaker_id=args.omit_speaker_id,
        pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor
    )

    print(f"\n✅ Done! Tokenized data saved to: {args.output}")
    print("🎉 Ready for training with sesame-finetune!")
