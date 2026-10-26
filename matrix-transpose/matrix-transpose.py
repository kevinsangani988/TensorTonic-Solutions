import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    a = np.array(A)
    a_transpose = a.T
    
    return a_transpose
