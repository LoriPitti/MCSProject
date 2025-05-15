#calcolo delle iterazioni e valutazione delle precisioni
import numpy  as np
from utils.iterResult import IterationResult
from utils.compute_res import ComputeResult
from lib.jacobi import jacobi
from lib.gauss_seidel import gauss_seidel
from lib.gradient  import gradient_method
from utils.error import relative_error
from utils.plot import plot_result, plot_matrix_heatmap, plot_difference
from utils.load import load_matrix
from utils.plot import plot_gradient_norm
from pathlib import Path
from scipy.sparse import csr_matrix


def compute_iterations(A: np.ndarray, b:np.ndarray, x:np.ndarray,tol:float, name, i:float)-> ComputeResult:

    max_iter = 30000

    #compute jacobi
    jacobi_res = jacobi(A,b,tol,max_iter)
    console_log_res(jacobi_res, "Jacobi", x)

    #compute gauss_seidel
    gauss_res = gauss_seidel(A,b,tol,max_iter)
    console_log_res(gauss_res, "Gauss-Seidel", x)

    #compute gauss_seidel
    gradient_res, grad_norm = gradient_method(A,b,tol,max_iter)
    console_log_res(gradient_res,   "Gradient", x)
    plot_gradient_norm(grad_norm, name, i)
    
    return ComputeResult(jacobi_res, gauss_res, gradient_res)


#compute for all the tollerances and for all the matrix from folder
def compute_iterations_multi(tols: np.ndarray,  A:csr_matrix,b:np.ndarray, x:np.ndarray, name):

    #call compute_iterations for all tollerances
    results: list[ComputeResult] = []
    for i in tols:
        results.append(compute_iterations(A,b,x,i, name, i))

    #Extract the results of the Jacobi method
    jacobi_res = [res.jacobi_res for res in results]
    plot_result(tols,jacobi_res, name, "Jacobi")
                #plot_difference(jacobi_res[0].solution, "matrx")
    
    gaus_res = [res.gaus_res for res in results]
    plot_result(tols,gaus_res, name, "Gauss-Seidel")

    gradient_res =[res.gradient_res for res in results]
    plot_result(tols, gradient_res, name,  "Gradient")


    #Extract the results of the b method


def compute_matrix(tols: np.ndarray):
    folder_path = Path('matrix')
    for file_path in folder_path.iterdir():
       if file_path.is_file():
            A = load_matrix(file_path)
            #A = load_matrix('matrix/spa1.mtx')
            rows, cols  = A.shape
            print(f"Rows: {rows} Cols: {cols}")
            x = np.ones(cols) #create x vector = 1
            b = A @ x 
            plot_matrix_heatmap(A,file_path.stem)
            compute_iterations_multi(tols, A, b,x, file_path.stem)


def console_log_res(result: IterationResult, function, x:np.ndarray):
     if result.converged:
        print(f"\n{function} converged with iterations: {result.iterations} , time: {result.time_taken}, result:  {result.solution}")
        rel_error = round(relative_error(x, result.solution),8)
        result.rel_error = rel_error
        print(f"realtive error: {rel_error}")