import numpy as np

def _tanimoto_similarity(A: np.ndarray, B: np.ndarray):
    """
    Compute the pairwise tanimoto between matrices A and B

    :param A: (n, p) Fingerprint matrix of n samples
    :param B: (m, p) Fingerprint matrix of m samples
    :returns: (n, m) matrix with tanimoto similarity between A[i, :] and B[j, :] at index i,j
    """
    assert A.dtype is np.dtype(np.bool_)
    assert B.dtype is np.dtype(np.bool_)
    
    # Compute intersection using broadcasting
    # A.shape: (n, p) -> (n, 1, p)
    # B.shape: (m, p) -> (1, m, p)
    A_expanded = A[:, np.newaxis, :]  # (n, 1, p)
    B_expanded = B[np.newaxis, :, :]  # (1, m, p)
    
    # Logical AND for intersection
    A_intersection_B = np.bitwise_and(A_expanded, B_expanded)  # (n, m, p)
    masked_a_int_b = A_intersection_B.sum(axis=2)  # (n, m)
    
    # Sum along feature dimension
    sum_A = np.sum(A, axis=1)  # (n,)
    sum_B = np.sum(B, axis=1)  # (m,)
    
    # Broadcast sums for union calculation
    broadcast_A = sum_A[:, np.newaxis]  # (n, 1)
    broadcast_B = sum_B[np.newaxis, :]  # (1, m)
    
    # Compute Tanimoto similarity
    tanimoto_similarity = masked_a_int_b / (broadcast_A + broadcast_B - masked_a_int_b)
    
    return tanimoto_similarity