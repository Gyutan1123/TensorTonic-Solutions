import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """   
    try:
        matrix = np.array(matrix)
    except ValueError:
        return None

    if matrix.ndim != 2:
        return None
    
    if matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues, _ = np.linalg.eig(matrix)

    eigenvalues.sort()

    return eigenvalues