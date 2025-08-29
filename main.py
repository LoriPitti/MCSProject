from lib.jacobi import jacobi as j
import numpy as np
from  compute  import *

from  utils.iterResult import IterationResult
from utils.compute_res import ComputeResult
from scipy.io import mmread



# --- SELECT MODE ------
mode = input("Inserisci 1 per computare UNA sola matrice, 0 per computarle tutte: ").strip()

if mode == "1":
    # Parametri per una singola matrice
    file_path = Path('matrix/single_matrix/test_matrix.mtx')

    A = mmread(file_path).tocsr()
    rows, cols  = A.shape
    x = np.ones(cols)
    b = A @ x 
    tol = 1e-4

    plot_matrix_heatmap(A, file_path.stem)

    compute([tol], A, b, x, file_path.stem)

elif mode == "0":
    # Parametri per tutte le matrici
    tols = np.array([1e-3, 1e-5, 1e-7, 1e-9], dtype=float)
    load_and_compute_matrix(tols)

else:
    print("Scelta non valida. Inserisci 1 o 0.")





