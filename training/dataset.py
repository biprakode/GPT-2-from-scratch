import torch
from tqdm import tqdm

from tokenizer.bpe import BPETokenizer
import os
import pandas as pd
import numpy as np
from pathlib import Path

class TextDataset:
    def __init__(self , data_path:str, block_size:int = 1024):
        self.data_path = data_path
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.num_samples = (len(self.data) - 1) // block_size


    def __len__(self):
        return self.num_samples

    def __getitem__(self , idx):
        chunk = self.data[idx*self.block_size:(idx+1)*self.block_size + 1] # blocksize + 1 tokens
        input_ids = chunk[:-1] # first block - 1 tokens
        label = chunk[1:] # last block tokens shifted by 1
        input_id = torch.from_numpy(input_ids.astype(np.int64))
        label = torch.from_numpy(label.astype(np.int64))

        return {
            "input_id" : input_id,
            "label" : label
        }
