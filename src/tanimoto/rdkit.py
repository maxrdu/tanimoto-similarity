import numpy as np
from rdkit.DataStructs import cDataStructs


def _tanimoto_similarity(A, B):
    """
    Compute the pairwise tanimoto between matrices A and B

    :param A: (n, p) Fingerprint matrix of n samples
    :param B: (m, p) Fingerprint matrix of m samples
    :returns: (n, m) matrix with tanimoto similarity between A[i, :] and B[j, :] at index i,j
    """
    n = len(A)
    m = len(B)
    if m < n:
        # Might be faster to loop over the smaller array,
        sims = _tanimoto_similarity(B, A)  # (m, n)
        return sims.T


    sims = np.zeros((n, m))
    for i, query in enumerate(A):
        # Similarities to all entries in B, list of size m
        similarities = cDataStructs.BulkTanimotoSimilarity(query, B)
        sims[i, :] = np.array(similarities)

    return sims

