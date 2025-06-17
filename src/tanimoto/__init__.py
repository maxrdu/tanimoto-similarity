from typing import Literal, List

from torch import compile

from .torch import _tanimoto_similarity as torch_tanimoto_similarity
from .torch import _tanimoto_similarity_dot as torch_tanimoto_similarity_dot
from .rdkit import _tanimoto_similarity as rdkit_tanimoto_similarity
from .numpy import _tanimoto_similarity as numpy_tanimoto_similarity
from .numpy import _tanimoto_similarity_dot as numpy_tanimoto_similarity_dot
from .numba import _tanimoto_similarity as numba_tanimoto_similarity


_torch_tanimoto_similarity_cmp = compile(torch_tanimoto_similarity)


def tanimoto_similarity(
        A: List,
        B: List,
        method: Literal["torch", "rdkit", "numpy", "numpy-dot", "scipy", "numba", "torch-compile", "torch-dot"] = "rdkit",
        convert: bool = True
    ):
    """
    Compute the pairwise tanimoto between matrices A and B

    :param A: (n, p) List of n fingerprints
    :param B: (m, p) List of m fingerprints
    :returns: (n, m) matrix with tanimoto similarity between A[i, :] and B[j, :] at index i,j
    """

    if convert and method in ["numpy", "scipy", "numba"]:
        import numpy as np
        A = np.array([list(fp) for fp in A], dtype=np.bool_)
        B = np.array([list(fp) for fp in B], dtype=np.bool_)
    elif convert and method == "numpy-dot":
        import numpy as np
        A = np.array([list(fp) for fp in A], dtype=np.float32)
        B = np.array([list(fp) for fp in B], dtype=np.float32)
    elif convert and method in ["torch", "torch-compile"]:
        import torch
        A = torch.tensor([list(fp) for fp in A]).bool()
        B = torch.tensor([list(fp) for fp in B]).bool()
    elif convert and method == "torch-dot":
        import torch
        A = torch.tensor([list(fp) for fp in A])
        B = torch.tensor([list(fp) for fp in B])

    if method == "rdkit":
        return rdkit_tanimoto_similarity(A, B)
    elif method == "torch":
        return torch_tanimoto_similarity(A, B)
    elif method == "torch-compile":
        return _torch_tanimoto_similarity_cmp(A, B)
    elif method == "torch-dot":
        return torch_tanimoto_similarity_dot(A, B)
    elif method == "numpy":
        return numpy_tanimoto_similarity(A, B)
    elif method == "numpy-dot":
        return numpy_tanimoto_similarity_dot(A, B)
    elif method == "scipy":
        from scipy.spatial.distance import cdist
        return 1 - cdist(A, B, metric="jaccard")
    elif method == "numba":
        return numba_tanimoto_similarity(A, B)

    else:
        raise ValueError(f"Invalid method {method}.")