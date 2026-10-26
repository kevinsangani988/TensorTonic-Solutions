import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len

    result = np.full((len(seqs), L), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        result[i, :min(len(seq), L)] = seq[:L]

    return result
