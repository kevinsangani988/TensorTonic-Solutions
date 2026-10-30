import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    if len(y)==0:
        return 0.0

    _ , count = np.unique(y, return_counts=True)

    probabilities = count/len(y)

    return -np.sum(probabilities*np.log2(probabilities)).astype(float)