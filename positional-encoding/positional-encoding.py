import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    pos = np.array([i for i in range(seq_len)]).reshape(-1, 1)
    pos = pos.reshape(-1, 1)
    freq = np.array([base ** (-2*i / d_model) for i in range((d_model+1) // 2)])
    freq = freq.reshape(1, -1)

    PE = np.zeros((seq_len, d_model))

    if d_model % 2 == 0:
        PE[:, 0::2] = np.sin(pos * freq)
        PE[:, 1::2] = np.cos(pos * freq)
    else:
        PE[:, 0::2] = np.sin(pos * freq)
        PE[:, 1::2] = np.cos(pos * freq[:, :-1])

    return PE
