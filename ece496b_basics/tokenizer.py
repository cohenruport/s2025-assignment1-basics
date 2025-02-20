class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        for token in self.special_tokens:
            if token.encode('utf-8') not in self.vocab.values():
                self.vocab[len(self.vocab)] = token.encode('utf-8')
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.merge_dict = {b1 + b2: (b1, b2) for b1, b2 in self.merges}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = {int(line.split()[0]): line.split()[1].encode('utf-8') for line in f}
        
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges = [tuple(line.strip().split()) for line in f]
            merges = [(b1.encode('utf-8'), b2.encode('utf-8')) for b1, b2 in merges]
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens = self._tokenize(text)
        token_ids = []
        
        for token in tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes in self.id_to_token:
                token_ids.append(self.id_to_token[token_bytes])
            else:
                token_ids.append(self.id_to_token.get(b'<UNK>', -1))
        
        return token_ids

    def _tokenize(self, text: str) -> list[str]:
        tokens = list(text)  
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            bigrams = [b1 + b2 for b1, b2 in pairs]
            merge_candidates = [(i, bigram) for i, bigram in enumerate(bigrams) if bigram in self.merge_dict]
            
            if not merge_candidates:
                break
            
            i, bigram = merge_candidates[0]
            tokens = tokens[:i] + [bigram] + tokens[i + 2:]
        
        return tokens

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab.get(i, b'').decode('utf-8') for i in ids]
        return ''.join(tokens)

