import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
    
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)

    if X.ndim == 1:
        X = np.expand_dims(X, 0)
    y = np.array(y)

    W = np.zeros_like(X[0])
    b = 0

    for _ in range(steps):
        p = _sigmoid(W @ X.T + b)

        W = W - lr * np.mean((p-y) * X.T)
        b = b - lr * np.mean(p-y)

    return W, b
