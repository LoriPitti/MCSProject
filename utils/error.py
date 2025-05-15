#COMPUTES THE RELATIVE RESIDUAL ERROR ||Ax_k - b|| / ||b||: 
import numpy as np
from scipy.sparse import spmatrix

def compute_error(A,x_k: np.ndarray,b:  np.ndarray)->float:
    residual = A @ x_k -b
    num = np.linalg.norm(residual) #normmalization
    den = np.linalg.norm(b)
    return num/den



def relative_error(x_exact: np.ndarray, x_approx: np.ndarray) -> float:
    numerator = np.linalg.norm(x_exact - x_approx)
    denominator = np.linalg.norm(x_exact)
    return numerator / denominator
