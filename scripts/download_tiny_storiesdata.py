import os
from datasets import load_dataset

os.makedirs('data/raw', exist_ok=True)

dataset = load_dataset('roneneldan/TinyStories')

print("Writing train split...")
with open('data/raw/tinystories_train.txt', 'w') as f:
    for example in dataset['train']:
        f.write(example['text'] + '\n')

print("Writing validation split...")
with open('data/raw/tinystories_val.txt', 'w') as f:
    for example in dataset['validation']:
        f.write(example['text'] + '\n')

print("Tiny stories done.")