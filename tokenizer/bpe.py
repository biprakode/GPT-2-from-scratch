import json
import re


# GPT-2 word-splitting pattern: applied to raw text BEFORE byte encoding.
WORD_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+""")


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def get_byte_encoder() -> dict[int, str]:
    byte_str = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    cs = byte_str[:]
    n = 0
    for b in range(256):
        if b not in byte_str:
            byte_str.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(n) for n in cs]
    return dict(zip(byte_str, cs))


def byte_encode_word(word: str, byte_encoder: dict[int, str]) -> str:
    """Byte-encode a single word (already split from text) into BPE-ready chars."""
    return ''.join(byte_encoder[b] for b in word.encode('utf-8'))


class BPETokenizer:
    def __init__(self):
        self.vocab = {}           # id -> token
        self.inverse_vocab = {}   # token -> id
        self.merges = []
        self.bpe_ranks = {}
        self.cache = {}
        self.byte_encoder = get_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def load_vocab_merges(self, vocab_file: str, merge_file: str):
        with open(vocab_file, 'r') as f:
            loaded_vocab = json.load(f)
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        with open(merge_file, 'r') as f:
            lines = f.read().strip().split('\n')
            # skip version header if present
            if lines[0].startswith('#'):
                lines = lines[1:]
            for rank, line in enumerate(lines):
                pair = tuple(line.split())
                self.merges.append(pair)
                self.bpe_ranks[pair] = rank

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:  # no more known merges
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        # 1. Split raw text into words FIRST
        # 2. Byte-encode each word separately
        # 3. Apply BPE merges to each byte-encoded word
        words = WORD_PATTERN.findall(text)

        bpe_tokens = []
        for word in words:
            encoded_word = byte_encode_word(word, self.byte_encoder)
            bpe_result = self._bpe(encoded_word)
            for t in bpe_result.split(' '):
                bpe_tokens.append(self.inverse_vocab[t])

        return bpe_tokens

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.vocab[idx] for idx in token_ids]
        text = ''.join(tokens)

        byte_array = bytearray([self.byte_decoder[c] for c in text])
        return byte_array.decode('utf-8', errors='replace')
