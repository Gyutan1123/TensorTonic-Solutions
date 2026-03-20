def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation 
    """

    recommended = recommended[:k]

    recommended = set(recommended)
    relevant = set(relevant)

    hit = recommended & relevant

    return [len(hit) / k, len(hit) / len(relevant)]
    

    
    # Write code here