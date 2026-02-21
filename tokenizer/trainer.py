import json
from collections import Counter
from tqdm import tqdm

from tokenizer.bpe import BPETokenizer, WORD_PATTERN, get_byte_encoder, get_pairs, byte_encode_word


class BPE_Trainer:
    def __init__(self, num_merges: int = 50000):
        self.num_merges = num_merges
        self.byte_encoder = get_byte_encoder()

    def train(self, corpus: str):
        # Split raw text into words FIRST, then byte-encode each word
        raw_words = WORD_PATTERN.findall(corpus)
        encoded_words = [byte_encode_word(w, self.byte_encoder) for w in raw_words]

        # Build frequency table: each word as space-separated chars
        word_freqs = Counter(' '.join(w) for w in encoded_words)

        merges = []
        bpe_ranks = {}

        for rank in tqdm(range(self.num_merges), desc="BPE merges"):
            # Count all adjacent pairs across the vocabulary
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                pairs = get_pairs(symbols)
                for pair in pairs:
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Find best pair
            best_pair = max(pair_counts, key=pair_counts.get)
            bpe_ranks[best_pair] = rank
            merges.append(best_pair)

            # Apply merge to every word in word_freq
            bigram_str = ' '.join(best_pair)
            merged_str = ''.join(best_pair)
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram_str, merged_str)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

        # Build vocab: 256 base byte chars + one new token per merge
        inverse_vocab = {}  # token -> id
        idx = 0
        for byte_char in self.byte_encoder.values():
            if byte_char not in inverse_vocab:
                inverse_vocab[byte_char] = idx
                idx += 1
        # Each merge creates exactly one new token
        for pair in merges:
            merged_token = ''.join(pair)
            if merged_token not in inverse_vocab:
                inverse_vocab[merged_token] = idx
                idx += 1

        vocab = {v: k for k, v in inverse_vocab.items()}

        tokenizer = BPETokenizer()
        tokenizer.vocab = vocab
        tokenizer.inverse_vocab = inverse_vocab
        tokenizer.merges = merges
        tokenizer.bpe_ranks = bpe_ranks

        print(f"Training complete. Vocab size: {len(vocab)}, Merges: {len(merges)}")
        return tokenizer

    def save(self , tokenizer: BPETokenizer, vocab_file: str = 'data/vocab/my_vocab.json', merge_file: str = 'data/vocab/my_merges.txt'):
        out_vocab = {token: idx for idx, token in tokenizer.vocab.items()}
        with open(vocab_file, 'w') as f:
            json.dump(out_vocab, f, ensure_ascii=False)

        with open(merge_file, 'w') as f: # store tokenizer merges
            f.write('#version: 0.2\n')
            for pair in tokenizer.merges:
                f.write(f'{pair[0]} {pair[1]}\n')