from pyomo.environ import * 
from pyomo.mpec import *
import numpy as np 
import pandas as pd
import os, time
from numpy import linalg
from typing import Any, Dict, Self
from pyomo.core.expr.numeric_expr import SumExpression

class LcpDCSolver: 
    """
    Solver for Linear Complementarity Problems using the DC Algorithm (DCA) 
    for bilinear terms (DCA-BL).

    Attributes:
        name (str): Name of the model.
        model (ConcreteModel): Pyomo ConcreteModel for the LCP.
        instance (ConcreteModel): Instance of the model.
        x_init (np.ndarray): Starting point
        N_mat_inv (np.ndarray): Inverse of the linear transformation matrix N.
        M1 (np.ndarray): The M1 matrix in the LCP formulation.
        c1 (np.ndarray): The c1 vector in the LCP formulation.
        sparse (bool): Indicates if the M1 matrix is sparse.
    """
    def __init__(self: Self, name: str, sparse: bool=False):
        """
        Initializes the LcpDCSolver with a given name and sparsity option.

        Args:
            name (str): Name of the model.
            sparse (bool): Indicates if the M1 matrix is sparse.
        """
        self.name: str = name
        self.model: ConcreteModel = None
        self.instance: ConcreteModel = None
        self.x_init: np.ndarray = None
        self.N_mat_inv: np.ndarray = None
        self.M1: np.ndarray = None
        self.c1: np.ndarray = None
        self.sparse:bool = sparse
    

    def constructLCP(
            self: Self, 
            N: int,  
            M1: np.ndarray, 
            c1: np.ndarray, 
            N_mat: np.ndarray, 
            alpha: np.ndarray, 
            beta: np.ndarray, 
        ) -> None:
        """
        Constructs the Pyomo model for the LCP using DCA-BL.

        Args:
            N (int): Dimension of the LCP.
            M1 (np.ndarray): The M coefficient matrix in the LCP formulation.
            c1 (np.ndarray): The c1 rhs vector in the LCP formulation.
            N_mat (np.ndarray): The linear transformation matrix N.
            alpha (np.ndarray): The alpha parameters for the bilinear terms.
            beta (np.ndarray): The beta parameters for the bilinear terms.
        """

        self.N = N
        model = ConcreteModel(name=self.name)
        model.dual = Suffix(direction=Suffix.IMPORT)
        if not self.sparse:
            # Initialize Model Parameters
            model.n = RangeSet(0,N-1)
            model.c1 = Param(model.n, initialize=dict(zip(range(N), c1)))
            model.N_mat_inv = Param([0,1],[0,1], initialize=self.np_to_pyo(linalg.inv(N_mat)))
            model.alpha = Param(model.n, initialize=dict(zip(range(N),alpha)))
            model.beta = Param(model.n, initialize=dict(zip(range(N),beta)))

            model.v_k = Param(model.n, initialize=dict(zip(range(N),np.zeros(N))), mutable=True)
            model.u_k = Param(model.n, initialize=dict(zip(range(N),np.zeros(N))), mutable=True)
            model.rho = Param(initialize=1, mutable=True)

            # Declare Variables
            model.y = Var(model.n, within=NonNegativeReals)
            model.x = Var(model.n, within=NonNegativeReals)
            model.t = Var(within=NonNegativeReals)
            model.u = Var(model.n, within=NonNegativeReals)
            model.v = Var(model.n, within=Reals)
            self.M1 = M1


        else:
            # Initialize Model Parameters
            model.n = RangeSet(0,N-1)
            self.M = M1
            M1_dok = M1.todok()

            model.N_mat_inv = Param([0,1],[0,1], initialize=self.np_to_pyo(linalg.inv(N_mat)))
            model.alpha = Param(model.n, initialize=dict(zip(range(N),alpha)))
            model.beta = Param(model.n, initialize=dict(zip(range(N),beta)))

            model.v_k = Param(model.n, initialize=dict(zip(range(N),np.zeros(N))), mutable=True)
            model.u_k = Param(model.n, initialize=dict(zip(range(N),np.zeros(N))), mutable=True)
            model.rho = Param(initialize=1, mutable=True)

            # Declare Variables
            model.y = Var(model.n, within=NonNegativeReals)
            model.x = Var(model.n, within=NonNegativeReals)
            model.t = Var(within=NonNegativeReals)
            model.u = Var(model.n, within=NonNegativeReals)
            model.v = Var(model.n, within=Reals)

            
        self.N_mat_inv = linalg.inv(N_mat)
        self.model = model
        self.N = N
        
        self.c1 = c1
        
        def M1_con(m: ConcreteModel, i: int) -> SumExpression:
            """
            LCP Inequality Constraint: M1*x + c1 = y
            0 <= y ⊥ x >= 0

            Args:
                m (ConcreteModel): The Pyomo model.
                i (int): Row index for the constraint.
            """ 
            
            expr = 0
            if self.sparse:
                nzi =  M1_dok.getrow(i).nonzero()[1]
                for j in nzi:
                    expr += M1_dok[i,j]*m.x[j]
                   
            else: 
                expr = sum(M1[i,j]*m.x[j] for j in m.n)
            
            return expr + c1[i] == m.y[i]

        def M2_con(m: ConcreteModel, i: int) -> SumExpression:
            """
            DCA-BL Majorization Constraint 1: 
                u^2 - v_k^2 - 2*v_k*(v - v_k) <= t
            
            Args:
                m (ConcreteModel): The Pyomo model.
                i (int): Row index for the constraint.
            """
            return m.alpha[i]*(m.u[i]**2) - m.beta[i]*(m.v_k[i]**2) - 2*m.beta[i]*m.v_k[i]*(m.v[i] - m.v_k[i]) <= m.t

        def M2_con2(m: ConcreteModel, i: int) -> SumExpression:
            """
            DCA-BL Majorization Constraint 2:
                v^2 - u_k^2 - 2*u_k*(u - u_k) <= t
            
            Args:
                m (ConcreteModel): The Pyomo model.
                i (int): Row index for the constraint.
            """
            return m.beta[i]*(m.v[i]**2) - m.alpha[i]*(m.u_k[i]**2) - 2*m.alpha[i]*m.u_k[i]*(m.u[i] - m.u_k[i]) <= m.t
        
        def u_con(m: ConcreteModel, i: int) -> SumExpression:
            """
            Linear Transformation Constraint for u:
                u = a * x + b * y
            Args:
                m (ConcreteModel): The Pyomo model.
                i (int): Index for the constraint.
            """
            return (m.N_mat_inv[0,0]*m.x[i] + m.N_mat_inv[0,1]*m.y[i]) == m.u[i]

        def v_con(m: ConcreteModel, i: int) -> SumExpression:
            """
            Linear Transformation Constraint for v:
                v = c * x + d * y
            Args:
                m (ConcreteModel): The Pyomo model.
                i (int): Index for the constraint.
            """
            return (m.N_mat_inv[1,0]*m.x[i] + m.N_mat_inv[1,1]*m.y[i]) == m.v[i]
        
        def penalty(m: ConcreteModel) -> SumExpression:
            """
            DCA-BL Penalty Objective Function:
                min rho*t
            Args:
                m (ConcreteModel): The Pyomo model.
            """
            return m.rho*m.t
        
        # Initialize Model Components
        model.obj = Objective(rule=penalty)
        model.M1_con = Constraint(model.n, rule=M1_con)
        model.M2_con = Constraint(model.n, rule=M2_con)
        model.M2_con2 = Constraint(model.n, rule=M2_con2)
        model.u_con = Constraint(model.n, rule=u_con)
        model.v_con = Constraint(model.n, rule=v_con)

        
        return model
        
    def initializeLCP(self, x_init,rho_init, t_init):
        """
        Initializes the LCP model instance with given starting values.

        Args:
            x_init (np.ndarray): Starting point for the variable x.
            rho_init (float): Initial penalty parameter.
            t_init (float): Initial value for the variable t.
        Returns:
            ConcreteModel: An instance of the initialized model.
        """
        sp = self.model.create_instance()
 
        sp.x.value = dict(zip(range(self.N), x_init))
        sp.t.value = t_init
        sp.rho.value = rho_init
        self.x_init = x_init
        
        return sp

    def np_to_pyo(self: Self, arr: np.ndarray) -> Dict[Any, Any]:
        """
        Converts a NumPy array to a Pyomo-compatible dictionary.
        Args:
            arr (np.ndarray): The NumPy array to convert.

        Returns:
            dict: A Pyomo-compatible dictionary representation of the array.
        """

        if len(arr.shape) == 1:
            return {i+1: arr[i] for i in range(arr.shape[0])}
        else:
            return{(i,j): arr[i,j] for i in range(arr.shape[0]) for j in range(arr.shape[1])}
    
    
    def f1(self,z):
        n = z.shape[0]
        Bz = (self.B@z).reshape(n)
        return 0.5*sum(Bz[i]*z[i] for i in range(self.N)) + sum(self.c1[i]*z[i] for i in range(self.N))[0]
    
    def computeComplError(self: Self, z: np.ndarray) -> float:
        """
        Computes the complementarity error for a given solution vector z.
        Args:
            z (np.ndarray): The solution vector.
        Returns:
            float: The complementarity error.
        """
        z_new = z.reshape(self.c1.shape)
        return np.abs(np.vdot(self.M@z_new + self.c1,z_new))

    def DCASolveLCP(
            self: Self, 
            sp: ConcreteModel, 
            solver: str, 
            eps: float, 
            delta1: float, 
            delta2: float, 
            max_iters: int, 
            verbosity: int = 1
        ) -> tuple[ConcreteModel, Any, int, float, float, bool]:
        """
        Main DCA-BL algorithm implementation of LCPs.

        Args:
            sp (ConcreteModel): The Pyomo model instance.
            solver (str): The solver to use ('gurobi', 'ipopt', etc
            eps (float): Convergence tolerance.
            delta1 (float): Parameter for updating the penalty.
            delta2 (float): Parameter for updating the penalty.
            max_iters (int): Maximum number of DCA-BL iterations.
            verbosity (int): Level of verbosity for logging (0: none, 1: iteration info).
        Returns:
            tuple: (ConcreteModel, SolverResults, int, float, float, bool)
                - ConcreteModel: The final model instance.
                - SolverResults: The results from the solver.
                - int: Number of iterations performed.
                - float: Total runtime.
                - float: Final penalty parameter value.
                - bool: Flag indicating if a solver warning occurred.
        """
        # Initialization
        opt = SolverFactory(solver)
        results = [] 
        x_k = self.x_init
        rho_k = sp.rho.value
        t_k = sp.t.value
        st = time.time()
        rt = 0
        results = []
        flag = False
        converged = False

        if verbosity == 1:
            print(f"============================ Running DCA Solve with {solver} ===================================")
            print(f"{'iter' : <10}{'x_diff': ^30}{'t':^30}{'rho':^10}")
        
        # Start DCA solve
        for k in range(max_iters):
            
            
            # Solve subproblem
            try:
                res = opt.solve(sp)
            
            except Exception:
                print(f"Solver Error on {k+1}.")
                if sp.rho.value > delta2:
                    sp.rho.value = sp.rho.value - delta2
                flag = True
                break
           

            if res.solver.status == SolverStatus.ok:
            
                x_kp1 = np.array([sp.x[i].value for i in sp.n])
                t_kp1 = sp.t.value
                rho_kp1 = sp.rho.value

                # Compute primal diff
                xdiff = linalg.norm(x_kp1 - x_k, ord=1)
                total_diff = xdiff

                # Dual Norms
                dual_norm = linalg.norm([sp.dual[sp.M1_con[i]] for i in sp.n], ord=1) 

                z_test = (total_diff < eps)
                t_test = (t_kp1 < eps)

                # Convergence check
                if t_test and z_test:
                    results.append([k+1, total_diff, t_kp1, rho_k])
                    converged = True
                    
                    if verbosity >0:
                        print(f"{k+1 : <10}{total_diff: ^30}{t_kp1:^30}{rho_k:^10}")
                    break
                
                # Update penalty
                r = min(1/total_diff, dual_norm + delta1) 
                if rho_kp1 >= r:
                    sp.rho.value = rho_k
                else:
                    sp.rho.value = rho_k + delta2
                
                x_k = x_kp1

                # update values 
                rho_k = sp.rho.value
                t_k = sp.t.value
                if solver == 'gurobi':
                    for i in sp.n:
                        if np.abs(sp.v[i].value) < 1e-4:
                            sp.v_k[i].value = 0
                        else:
                            sp.v_k[i].value = sp.v[i].value
                        if np.abs(sp.u[i].value) < 1e-4:
                            sp.u_k[i].value = 0
                        else:
                            sp.u_k[i].value = sp.u[i].value
                else: 
                    for i in sp.n:
                        sp.v_k[i].value = sp.v[i].value
                        sp.u_k[i].value = sp.u[i].value

                # store results
                results.append([k+1, total_diff, t_k, rho_k])
                if verbosity == 1:
                    print(f"{k+1 : <10}{total_diff: ^30}{t_k:^30}{rho_k:^10}")
                
            else: 
                print(f"Solver Warning on Iteration {k+1}.")
                if sp.rho.value > delta2:
                    sp.rho.value = sp.rho.value - delta2
                flag = True

        rt = time.time() - st

        if not converged:
            
            return sp, None, k+1, rt, rho_k, flag
        else: 
            return sp, res, k+1, rt, rho_k, flag
            
    