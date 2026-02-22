import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.dataset import TextDataset

dataset = TextDataset('data/tokenized/test.bin', block_size=128)
print(f"Dataset length: {len(dataset)}")

sample = dataset[0]
print(f"Input shape: {sample['input_id'].shape}")   # Should be (128,)
print(f"Labels shape: {sample['label'].shape}")     # Should be (128,)

# Verify alignment
assert sample['input_id'].shape[0] == 128
assert sample['label'].shape[0] == 128