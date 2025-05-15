from lib.jacobi import jacobi as j
import numpy as np
from  compute  import *

from  utils.iterResult import IterationResult
from utils.compute_res import ComputeResult



########################____________FUNCTION______________________####################
#compute_iterations(A,b,x,1e-4)
########################__________________________________________####################
if __name__ == '__main__':
    tols = np.array([10e-4,10e-6,10e-8,10e-10], dtype=float)
    compute_matrix(tols)



