import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.trainer import BPE_Trainer


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--corpus', type=str, default='data/raw/tinystories_train.txt',
                        help='Path to training corpus')
    parser.add_argument('--num-merges', type=int, default=50000,
                        help='Number of BPE merges')
    parser.add_argument('--max-chars', type=int, default=1000_000_000,
                        help='Max characters to read from corpus')
    parser.add_argument('--vocab-out', type=str, default='data/vocab/my_vocab.json')
    parser.add_argument('--merges-out', type=str, default='data/vocab/my_merges.txt')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.vocab_out), exist_ok=True)

    print(f"Reading corpus from {args.corpus}...")
    with open(args.corpus, 'r') as f:
        corpus = f.read(args.max_chars)
    corpus = corpus.replace('\n', ' ')
    print(f"Corpus size: {len(corpus):,} characters")

    trainer = BPE_Trainer(num_merges=args.num_merges)
    tokenizer = trainer.train(corpus)
    trainer.save(tokenizer, vocab_file=args.vocab_out, merge_file=args.merges_out)
    print(f"Saved vocab to {args.vocab_out} and merges to {args.merges_out}")


if __name__ == '__main__':
    main()