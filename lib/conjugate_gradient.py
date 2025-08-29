import numpy as np
import time
from numba import njit
from scipy.sparse import csr_matrix
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
def conjugate_gradient_numba(data, indices, indptr, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    r = b - csr_matvec(data, indices, indptr, x)
    p = r.copy()
    rs_old = np.dot(r, r)

    grad_norms = np.empty(max_iter)
    b_norm = np.linalg.norm(b)

    for k in range(max_iter):
        Ap = csr_matvec(data, indices, indptr, p)
        alpha = rs_old / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        grad_norm = np.linalg.norm(r)
        grad_norms[k] = grad_norm

        if grad_norm / b_norm < tol:
            return x, k + 1, True, grad_norms[:k+1]

        rs_new = np.dot(r, r)
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, False, grad_norms

# Wrapper per usare CSR scipy + raccogliere tempo e errori
def conjugate_gradient_method(A: csr_matrix, b: np.ndarray, tol: float, max_iter: int):
    start_time = time.time()
    x, k, converged, grad_norms = conjugate_gradient_numba(A.data, A.indices, A.indptr, b, tol, max_iter)
    error = compute_error(A, x, b)
    return IterationResult(converged, x, k, time.time() - start_time, error), grad_norms