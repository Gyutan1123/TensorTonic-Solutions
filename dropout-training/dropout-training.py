import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """

    x = np.array(x)

    if rng is None:
        rng = np.random

    mask = rng.random(size=x.shape) > p
    mask = mask.astype(np.int32) / (1-p)
    x = x * mask

    return x, mask