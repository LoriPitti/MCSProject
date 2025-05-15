#Solve the linear system Ax = b using the Jacobi iterative method on SPARSE MATRIX

import numpy as np
import time
from utils.error import compute_error
from utils.iterResult import IterationResult
from scipy.sparse import csr_matrix, diags
from multiprocessing import Pool

def jacobi(A: csr_matrix, b: np.ndarray, tol: float, max_iter: int) -> IterationResult:
    start_time = time.time()
    x = np.zeros_like(b, dtype=float)
    n = A.shape[0] #num solutions

    D = A.diagonal() #extract diagonal vector
    R = A - diags(D,format='csr') #compute R 

    
    for k in range(max_iter): #Iter whithin max iter 
        x_k = (b - R @ x)/D    #vettorializzato -> le matrici sparse non  sono ottimizzate per l  accesso riga per riga R[i,:]

        error = compute_error(A, x_k, b)

        if error < tol:
            stop_time = time.time()
            return IterationResult(True, x_k, k+1, stop_time - start_time, error)

        x = x_k

    stop_time = time.time()
    error = compute_error(A, x_k, b)
    return IterationResult(False, x, max_iter, stop_time - start_time, error)