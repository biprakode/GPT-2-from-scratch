import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.trainer import BPE_Trainer
from tokenizer.bpe import BPETokenizer

def read_corpus(path='data/raw/TinyStoriesV2-GPT4-train.txt'):
    with open(path, 'r') as f:
        corpus = f.read(100_000_000)
    return corpus

corpus = read_corpus()
corpus = corpus.replace('\n', ' ')
tokenizer_trainer = BPE_Trainer()
trained_tokenizer = tokenizer_trainer.train(corpus)
tokenizer_trainer.save(trained_tokenizer)