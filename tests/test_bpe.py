import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tokenizer.bpe import BPETokenizer

custom_tokenizer = BPETokenizer()
custom_tokenizer.load_vocab_merges("/run/media/biprarshi/COMMON/files/AI/MyGPT-2/data/vocab/my_vocab.json", "/run/media/biprarshi/COMMON/files/AI/MyGPT-2/data/vocab/my_merges.txt")

official_tokenizer = BPETokenizer()
official_tokenizer.load_vocab_merges("/run/media/biprarshi/COMMON/files/AI/MyGPT-2/data/vocab/vocab.json", "/run/media/biprarshi/COMMON/files/AI/MyGPT-2/data/vocab/merges.txt")

def compare_tokenization(text):
    custom_ids = custom_tokenizer.encode(text)
    official_ids = official_tokenizer.encode(text)

    # Get the actual tokens
    custom_tokens = [custom_tokenizer.vocab[i] for i in custom_ids]
    official_tokens = [official_tokenizer.vocab[i] for i in official_ids]

    print(f"\n--- Comparison for: '{text}' ---")
    print(f"{'Model':<10} | {'Count':<5} | {'Tokens'}")
    print("-" * 60)
    print(f"{'Custom':<10} | {len(custom_ids):<5} | {custom_tokens}")
    print(f"{'Official':<10} | {len(official_ids):<5} | {official_tokens}")

compare_tokenization("The quick brown fox jumps over the lazy dog.")
compare_tokenization("Transformer architectures utilize multi-head attention mechanisms.")