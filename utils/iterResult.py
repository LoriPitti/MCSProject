#Class result with solution, iteration, time taken 
import numpy as np

class IterationResult:
    def __init__(self, converged: bool, solution: np.ndarray, iterations: int, time_taken: float, rel_error: float):
        self.converged = converged
        self.solution = solution
        self.iterations = iterations
        self.time_taken = time_taken
        self.rel_error =  rel_error