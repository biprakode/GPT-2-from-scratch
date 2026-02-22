import os
from datasets import load_dataset
dataset = load_dataset('Skylion007/openwebtext')
dataset.to_parquet('data/raw/openwebtext.parquet')