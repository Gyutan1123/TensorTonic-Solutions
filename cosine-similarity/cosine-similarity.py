import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """

    a, b = np.array(a), np.array(b)
    if np.dot(a,b) == 0 :
        return 0
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
