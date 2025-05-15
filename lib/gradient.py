
import time
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, diags
from utils.error import compute_error
from utils.iterResult import IterationResult

@njit
def csr_matvec(data, indices, indptr, x):
    n = len(indptr) - 1
    result = np.zeros(n)
    for i in range(n):
        for idx in range(indptr[i], indptr[i + 1]):
            result[i] += data[idx] * x[indices[idx]]
    return result

@njit
def gradient_method_numba(data, indices, indptr, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    r = b - csr_matvec(data, indices, indptr, x)

    grad_norms = np.empty(max_iter)

    b_norm = np.linalg.norm(b)

    for k in range(max_iter):
        Ar = csr_matvec(data, indices, indptr, r)
        rr = np.dot(r, r)
        alpha = rr / np.dot(r, Ar)
        x = x + alpha * r
        r_new = r - alpha * Ar

        grad_norm = np.linalg.norm(r_new)
        grad_norms[k] = grad_norm

        if grad_norm / b_norm < tol:
            return x, k + 1, True, grad_norms[:k+1]

        r = r_new

    return x, max_iter, False, grad_norms


def gradient_method(A: csr_matrix, b: np.ndarray, tol: float, max_iter: int):
    start_time = time.time()
    x, k, converged, grad_norms = gradient_method_numba(A.data, A.indices, A.indptr, b, tol, max_iter)
    error = compute_error(A, x, b)
    return IterationResult(converged, x, k, time.time() - start_time, error), grad_norms