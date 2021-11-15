import numpy as np

class opt_solver():
    """
    Conform algorithms to project's optimisation output
    """
    def __init__(self,dim):
        self.eval = float('inf')
        self.xarray = np.empty(dim)
        self.n_iter = 0
        self.n_feval = 0
        self.success = False
        self.message = ''

    def set_message(self,error):
        dic_termination = {1:'Optimisation terminated successfully',2:'Number of maximum iterations exceeded',3: 'Number of maximum function evaluations exceeded'}
        self.message = dic_termination[error]

        if error == 1:
            self.success = True

def opt_converge(f_eval,tol):
    return np.std(f_eval) <= tol * np.abs(np.mean(f_eval))