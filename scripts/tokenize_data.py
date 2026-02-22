from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import pandas as pd
import numpy as np
from tokenizer.bpe import BPETokenizer

def tokenize_openwebtext(tokenizer, data_dir="data/raw"):
    all_tokens = []
    files = sorted(Path(data_dir).glob("*.parquet"))

    print(f"Found {len(files)} parquet files")

    for file in tqdm(files, desc="Tokenizing files"):
        print(f"\nProcessing {file.name}...")
        df = pd.read_parquet(file)
        print(f"  Loaded {len(df)} documents")

        for idx, text in enumerate(tqdm(df['text'], desc=f"  Tokenizing {file.name}", leave=False)):
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(50256)

            if idx % 1000 == 0:
                print(f"    Processed {idx} docs, {len(all_tokens)} tokens so far")

    print(f"\nTotal tokens before saving: {len(all_tokens)}")
    all_tokens = np.array(all_tokens , dtype=np.uint16)
    split_idx = int(len(all_tokens) * 0.9)
    train , val = all_tokens[:split_idx], all_tokens[split_idx:]
    train.tofile("data/tokenized/train.bin")
    val.tofile("data/tokenized/val.bin")
    print(f"Train tokens: {len(train)}")
    print(f"Val tokens: {len(val)}")


tokenizer = BPETokenizer()
tokenizer.load_vocab_merges("/run/media/biprarshi/COMMON/files/AI/MyGPT-2/data/vocab/vocab.json", "/run/media/biprarshi/COMMON/files/AI/MyGPT-2/data/vocab/merges.txt")

tokenize_openwebtext(
    data_dir='data/raw',
    tokenizer= tokenizer,
    output_file='data/tokenized/raw.bin'
)

