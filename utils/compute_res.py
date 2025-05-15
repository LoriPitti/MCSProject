
from utils.iterResult import IterationResult
class ComputeResult:
    def __init__(self, jacobi_res: IterationResult, gaus_res: IterationResult, gradient_res: IterationResult):
        self.jacobi_res = jacobi_res
        self.gaus_res = gaus_res
        self.gradient_res = gradient_res
        # self.g3 = g3