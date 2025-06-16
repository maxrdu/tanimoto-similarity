import torch
import numpy as np

def _tanimoto_similarity(A: torch.Tensor, B: torch.Tensor):
    """
    Compute the pairwise tanimoto between matrices A and B

    :param A: (n, p) Fingerprint matrix of n samples
    :param B: (m, p) Fingerprint matrix of m samples
    :returns: (n, m) matrix with tanimoto similarity between A[i, :] and B[j, :] at index i,j
    """
    assert A.dtype == torch.bool
    assert B.dtype == torch.bool

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    A = torch.tensor(np.stack(A)).to(device)
    B = B.to(device)
    n, m = A.shape[-2], B.shape[-2]

    A_intersection_B = torch.bitwise_and(A.unsqueeze(1), B.unsqueeze(0))
    masked_a_int_b = A_intersection_B
    masked_a_int_b = masked_a_int_b.sum(dim=2)

    sum_A = torch.sum(A, dim=1)
    sum_B = torch.sum(B, dim=1)
    broadcast_A = sum_A.unsqueeze(1).repeat(1, m)
    broadcast_B = sum_B.unsqueeze(1).repeat(1, n).T
    tanimoto_similarity = masked_a_int_b/(broadcast_A + broadcast_B - masked_a_int_b)
    return tanimoto_similarity