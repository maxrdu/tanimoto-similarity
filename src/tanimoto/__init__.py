from typing import Literal, List


from .torch import _tanimoto_similarity as torch_tanimoto_similarity
from .rdkit import _tanimoto_similarity as rdkit_tanimoto_similarity
from .numpy import _tanimoto_similarity as numpy_tanimoto_similarity



def tanimoto_similarity(A: List, B: List, method: Literal["torch", "rdkit", "numpy"] = "rdkit"):
    """
    Compute the pairwise tanimoto between matrices A and B

    :param A: (n, p) List of n fingerprints
    :param B: (m, p) List of m fingerprints
    :returns: (n, m) matrix with tanimoto similarity between A[i, :] and B[j, :] at index i,j
    """

    if method == "rdkit":
        return rdkit_tanimoto_similarity(A, B)
    
    elif method == "torch":
        import torch
        A = torch.tensor([list(fp) for fp in A]).bool()
        B = torch.tensor([list(fp) for fp in B]).bool()
        return torch_tanimoto_similarity(A, B)
    elif method == "numpy":
        import numpy as np
        A = np.array([list(fp) for fp in A], dtype=np.bool_)
        B = np.array([list(fp) for fp in B], dtype=np.bool_)
        return numpy_tanimoto_similarity(A, B)

    elif method == "scipy":
        from scipy.spatial.distance import cdist
        import numpy as np
        A = np.array([list(fp) for fp in A], dtype=np.bool_)
        B = np.array([list(fp) for fp in B], dtype=np.bool_)
        return 1 - cdist(A, B, metric="jaccard")

    else:
        raise ValueError(f"Invalid method {method}.")