

import numpy as np
import time
from utils.error import compute_error
from utils.iterResult import IterationResult
from scipy.sparse import csr_matrix, diags
from numba import njit


def gauss_seidel(A: csr_matrix, b: np.ndarray, tol: float, max_iter: int) -> IterationResult:
    start_time = time.time()
    x = np.zeros_like(b)
    
    for k in range(max_iter):
        x = gauss_seidel_single_iteration(A.data, A.indices, A.indptr, b, x)
        error = compute_error(A, x, b)
        if error < tol:
            return IterationResult(True, x, k + 1, time.time() - start_time, error)
    
    error = compute_error(A, x, b)
    return IterationResult(False, x, max_iter, time.time() - start_time, error)

@njit
def gauss_seidel_single_iteration(data, indices, indptr, b, x):
    n = len(b)
    x_new = x.copy()

    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]

        sum_ = 0.0
        a_ii = 0.0

        for idx in range(row_start, row_end):
            j = indices[idx]
            a_ij = data[idx]

            if j == i:
                a_ii = a_ij
            else:
                sum_ += a_ij * (x_new[j] if j < i else x[j])

        x_new[i] = (b[i] - sum_) / a_ii

    return x_new