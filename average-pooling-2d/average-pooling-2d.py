def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    h, w = len(X), len(X[0])
    p = pool_size
    hout, wout = h // p, w // p
    out = [[0]*wout for _ in range(hout)]

    for i in range(hout):
        for j in range(wout):
                for a in range(p):
                    for b in range(p):
                        out[i][j] += X[i*p+a][j*p+b] / (p**2)

    return out