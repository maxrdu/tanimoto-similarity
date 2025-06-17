import numba as nb
import numpy as np

@nb.njit
def popcount_builtin(x):
    """
    Hardware-optimized popcount using parallel bit counting
    Fastest for modern CPUs
    """
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    return (x * 0x0101010101010101) >> 56


@nb.njit
def pack_bool_matrix_optimized(bool_matrix):
    """
    Optimized boolean matrix packing with minimal overhead
    """
    n_rows, n_cols = bool_matrix.shape
    n_ints_per_row = (n_cols + 63) >> 6  # Fast division by 64
    
    packed_matrix = np.zeros((n_rows, n_ints_per_row), dtype=nb.uint64)
    
    for row in range(n_rows):
        for col in range(n_cols):
            if bool_matrix[row, col]:
                int_idx = col >> 6  # Fast division by 64
                bit_idx = col & 63  # Fast modulo 64
                packed_matrix[row, int_idx] |= nb.uint64(1) << nb.uint64(bit_idx)
    
    return packed_matrix


@nb.njit
def popcount_packed_row(packed_row):
    """Count total bits in a packed row"""
    total = 0
    for i in range(packed_row.shape[0]):
        total += popcount_builtin(packed_row[i])
    return total


@nb.njit
def intersection_count_packed_rows(packed_a, packed_b):
    """Count intersection bits between two packed rows"""
    count = 0
    min_len = min(packed_a.shape[0], packed_b.shape[0])
    for i in range(min_len):
        count += popcount_builtin(packed_a[i] & packed_b[i])
    return count


@nb.njit("float64[:,:](boolean[:,:], boolean[:,:])")
def _tanimoto_similarity(A, B):
    """
    Fastest Tanimoto similarity implementation for boolean matrices
    
    Args:
        A: Boolean matrix of shape (n_samples_A, n_features)
        B: Boolean matrix of shape (n_samples_B, n_features)
    
    Returns:
        Float64 matrix of shape (n_samples_A, n_samples_B) with Tanimoto similarities
    """
    n_a, n_features_a = A.shape
    n_b, n_features_b = B.shape
    
    # Ensure same number of features
    if n_features_a != n_features_b:
        raise ValueError("A and B must have the same number of features")
    
    # Pack boolean matrices to bit arrays
    A_packed = pack_bool_matrix_optimized(A)
    B_packed = pack_bool_matrix_optimized(B)
    
    # Pre-compute bit counts for all rows
    A_counts = np.empty(n_a, dtype=nb.int32)
    B_counts = np.empty(n_b, dtype=nb.int32)
    
    for i in range(n_a):
        A_counts[i] = popcount_packed_row(A_packed[i])
    
    for i in range(n_b):
        B_counts[i] = popcount_packed_row(B_packed[i])
    
    # Compute similarities
    sims = np.empty((n_a, n_b), dtype=nb.float64)
    
    for i in range(n_a):
        for j in range(n_b):
            intersection = intersection_count_packed_rows(A_packed[i], B_packed[j])
            union = A_counts[i] + B_counts[j] - intersection
            
            if union == 0:
                sims[i, j] = 1.0
            else:
                sims[i, j] = nb.float64(intersection) / nb.float64(union)
    
    return sims
