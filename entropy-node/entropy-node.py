import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable 
    """
    y = np.array(y)
    _, cnt = np.unique(y, return_counts=True)

    p = cnt / np.sum(cnt)

    return -np.sum(p*np.log2(p))