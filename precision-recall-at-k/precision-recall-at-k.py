def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    recommended = set(recommended[:k])
    relevant = set(relevant)
    precision = len(recommended&relevant)/ k
    recall = len(recommended&relevant)/ len(relevant)
    return [precision,recall]