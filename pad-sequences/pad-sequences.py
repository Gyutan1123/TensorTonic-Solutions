import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    for i in range(len(seqs)):
        seq = seqs[i]
        if len(seq) < max_len:
            seqs[i] = seq + [pad_value]*(max_len-len(seq))
        if len(seq) > max_len:
            seqs[i] = seq[:max_len]

    return np.array(seqs)