
import collections


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    
    for token in special_tokens:
        vocab[next_token_id] = token.encode('utf-8')
        next_token_id += 1
    
    with open(input_path, 'rb') as f:
        text = f.read()
    
    sequences = [list(chunk) for chunk in [text]]
    merges = []
    
    def get_pair_counts(sequences):
       pair_counts = collections.defaultdict(int)
       for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (bytes([seq[i]]), bytes([seq[i + 1]]))
                pair_counts[pair] += 1
       return pair_counts

    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(sequences)

        if not pair_counts:
           break


        best_pair = max(pair_counts, key=pair_counts.get)

        new_token = best_pair[0] + best_pair[1]
        vocab[next_token_id] = new_token
        merges.append(best_pair)
        next_token_id += 1
        
        new_sequences = []
        for seq in sequences:
            i = 0
            new_seq = []
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(new_seq)
        sequences = new_sequences
        
        if len(vocab) >= vocab_size:
            break
    
    return vocab, merges
