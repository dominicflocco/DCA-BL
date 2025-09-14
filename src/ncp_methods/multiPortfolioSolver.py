from pyomo.environ import * 
from pyomo.mpec import *
import pyomo.kernel as pmo
import numpy as np 
import pandas as pd
import os,time
import gurobipy as gp
from gurobipy import GRB	
from numpy import linalg
import scipy as sc
from itertools import product
from typing import Any, Dict, Self

class multiPorfolioSolver:
    """
    Class to solve multi-portfolio equilibrium problems using DCA-BL
    with long-only positions

    Attributes:
        data (Dict[str, Any]): Dictionary containing the problem instance data.
        portfolio (Dict[str, Any]): Dictionary containing portfolio data.
        firm (Dict[str, Any]): Dictionary containing firm data.
        firms (range): Range object representing the number of firms.
        mu (np.ndarray): Matrix of expected returns for each portfolio and asset.
        V (np.ndarray): Covariance matrix of asset returns.
        B (Dict[int, float]): Dictionary of budget constraints for each firm.
        R_min (Dict[int, float]): Dictionary of minimum return requirements for each firm.
        f_inv (Dict[int, float]): Dictionary of risk tolerance parameters for each firm.
        mat (Dict[Tuple[int, int], np.ndarray]): Dictionary of interaction matrices for each firm and portfolio.
        assets (range): Range object representing the number of assets.
        portfolios (range): Range object representing the number of portfolios.
        idx (Dict[Tuple[int, int], int]): Dictionary mapping portfolio-asset pairs to indices.
        f_map (Dict[Tuple[int, int, int], int]): Dictionary mapping firm-portfolio-asset triples to indices.
        constr_to_update (list): List of constraints to update in the DCA-BL algorithm.
        vars_to_update (list): List of variable tuples to update in the DCA-BL algorithm.
        N_mat (np.ndarray): Matrix used for bilinear transformation.
        alpha (float): Parameter for bilinear DC decomposition.
        beta (float): Parameter for bilinear DC decomposition.
    """

    def __init__(self: Self, inst: Dict[str, Any]) -> None:
        """
        Initialize the multiPortfolioSolver with the given instance data.
        """
        self.data = inst 
        self.portfolio = self.data['portfolio']
        self.firm = self.data['firm']
        self.firms = range(self.firm['numFirms'])
        self.mu = np.vstack([self.portfolio[p]['mu'] for p in self.portfolio.keys()])
        self.V = self.portfolio[len(self.portfolio) - 1]['V']

        self.B = {f:self.firm[f]['B'] for f in self.firms}
        self.R_min = {f:self.firm[f]['R'] for f in self.firms}
        self.f_inv = {f:self.firm[f]['F_inv'] for f in self.firms}
        self.mat = {(f,p):self.firm[f]['mat'][p] for p in self.portfolio.keys() for f in self.firms}
        self.assets = range(len(self.portfolio[0]['assets']))
        self.portfolios = range(len(self.portfolio.keys()))
        self.firms = range(len(self.firms))
        idx,f_map = {}, {}
        j = 0
        for f in self.firms:
            i = 0
            for p in self.portfolios:
                for a in self.assets:
                    idx[(p,a)] = i
                    f_map[(f,p,a)] = j
                    i += 1
                    j+=1
        self.idx = idx 
        self.f_map = f_map 

        self.constr_to_update = []
        self.vars_to_update = []
        self.N_mat=np.array([[1,1],[1,-1]])
        self.alpha=1
        self.beta=1

    def f(self: Self, x: gp.Var, f: int) -> gp.QuadExpr:
        """
        Objective function for firm f.
        Args:
            x_long (gp.Var): Gurobi variable for long positions.
            x_short (gp.Var): Gurobi variable for short positions.
            f (int): Index of the firm.
        Returns:
            gp.QuadExpr: The objective function expression for firm f.
        """
        obj = -sum(self.B[f]*self.mu[p,a]*x[f,p,a].X for p,a in product(self.portfolios, self.assets))
        for p in self.portfolios:
            temp_sum = [sum(self.B[ff]*x[ff,p,a].X for ff in self.firms) for a in self.assets]
            obj += sum(self.mat[f,p][a,aa]*temp_sum[a]*temp_sum[aa] for aa, a in product(self.assets,self.assets))
        return obj

    def grad_f(self: Self, x: gp.Var, f: int) -> list[gp.QuadExpr]:
        """
        Gradient of the objective function for firm f with respect to variables x.
        Args:
            x (gp.Var): Gurobi variable for long positions.
            f (int): Index of the firm.
        Returns:
            List[gp.QuadExpr]: The gradient of the objective function for firm f.
                - Element i corresponds to the partial derivative with respect to x[i].
        """
        gradf = [gp.quicksum(2*[self.B[ff]*self.mat[f,p][aa,a]*x[ff,p,a] for a, ff in product(self.assets, self.firms)]) - self.B[f]*self.mu[p,aa]  for p, aa in product(self.portfolios,self.assets)]
        return gradf


    def solveKKT(self: Self) -> dict[str, Any]:
        """
        Formulates and solves the multi-portfolio equilibrium problem as a
        nonconvex bilinear program using Gurobi's nonconvex solver.

        Returns:
            dict: A dictionary containing the solution details:
                - 'error': Complementarity error of the solution.
                - 'obj_vals': Objective values for each firm.
                - 'status': Gurobi solver status code.
                - 'rt': Runtime of the solver in seconds.
                - 'sol': The Gurobi model instance with the solution.
        """

        
        evals,U = sc.linalg.eigh(self.V)
        D = np.diag(np.sqrt(np.real(evals)))
        covSqrt = U@D@U.T

        m = gp.Model()

        m.params.NonConvex = 2
        m.params.OutputFlag = 1
        ##### VARIABLES ####

        ub = len(self.firms)*len(self.portfolios)*len(self.assets)
        # Primary Variables
        x = m.addVars(self.firms,self.portfolios,self.assets,lb=0,ub=ub,name='x')
       

        # Dual Variables 
        lmb = m.addVars(self.firms,lb=0,name='lmb')
        #gma = m.addVars(self.firms,lb=-np.inf,name='gma')
        gma = m.addVars(self.firms,self.portfolios,lb=-np.inf,name='gma')


        # Complementarity Variables
        lmb_c = m.addVars(self.firms,lb=0,name='lmb_c')
        #gma_c = m.addVars(self.firms,lb=0,name='gma_c')
        x_c = m.addVars(self.firms,self.portfolios,self.assets,lb=0,name='x_c')

        # Auxillary Variables
        std = m.addVars(self.firms,name='std')
        Vx = m.addVars(self.firms,self.portfolio,self.assets,lb=0, name='Vx')

        z = m.addVars(self.firms,self.portfolios,self.assets,lb=-np.inf,name='z')
        y = m.addVars(self.firms,self.portfolios,self.assets,lb=-np.inf,name='y')
        xi = m.addVars(self.firms,self.portfolios,self.assets,lb=-np.inf,name='xi')
        eta = m.addVars(self.firms,self.portfolios,self.assets,lb=-np.inf,name='eta')

        
        ###### CONSTRAINTS ######

       
        for f in self.firms:

            # Linear Constraints 
            m.addConstrs((gp.quicksum([x[f,p,a] for a in self.assets]) == 1 for p in self.portfolios),
                        name='budget')


    
            # Defining Constraints
            m.addConstrs(
                        gp.quicksum(
                                    [
                                        covSqrt[self.idx[pp,aa],self.idx[p,a]]*x[f,p,a] 
                                        for p, a in product(self.portfolios, self.assets)
                                    ]
                                    ) == Vx[f,pp,aa] 
                        for pp,aa in product(self.portfolios,self.assets)
                        )
            m.addConstrs((z[f,p,a] == self.grad_f(x,f)[self.idx[p,a]] + gma[f,p] 
                          for p,a in product(self.portfolios, self.assets)),
                          name='z_def')
            
            m.addConstrs((y[f,p,a] == self.mu[p,a]*std[f] 
                                        - self.f_inv[f]*gp.quicksum([self.V[self.idx[p,a],self.idx[pp,aa]]*x[f,pp,aa] 
                                                                     for pp,aa in product(self.portfolios,self.assets)])  
                         for p,a in product(self.portfolios, self.assets)),
                         name='y_def')
            
            
            m.addConstr(gp.quicksum([self.mu[p,a]*x[f,p,a] for p,a in product(self.portfolios,self.assets)]) + self.f_inv[f]*std[f] - self.R_min[f] == lmb_c[f],
                        name='lmb_c_def')
            
            
            # # Quadratic Constraints
            m.addQConstr(gp.quicksum([Vx[f,p,a]*Vx[f,p,a] for p,a in product(self.portfolios, self.assets)])==std[f]**2, name='var_convex')
        
    
            m.addConstrs((x_c[f,p,a] == xi[f,p,a] - eta[f,p,a] for p,a in product(self.portfolios,self.assets)),
                         name='x_c_def')

            m.addConstrs((xi[f,p,a] == std[f]*z[f,p,a] for p,a in product(self.portfolios,self.assets)),
                          name='xi_def')
            m.addConstrs((eta[f,p,a] == lmb[f]*y[f,p,a] for p,a in product(self.portfolios,self.assets)),
                         name='eta_def')

            # Complementarity constraints
            m.addConstrs((x_c[f,p,a]*x[f,p,a] == 0 for p,a in product(self.portfolios,self.assets)))
            m.addConstr(lmb_c[f]*lmb[f] == 0)
            
        
        m.setObjective(0, GRB.MINIMIZE)   

        m.write('test.lp')
        st = time.time()
       
        m.setParam('TimeLimit',20*60)
        m.optimize()
        
        rt = time.time() - st
        
        status = m.getAttr("Status")
        if status == 2:
            error = self.complError(m)
            obj_val = {f:self.f(x,f) for f in self.firms}
        else: 
            obj_val = {f:-1 for f in self.firms}
            error = -1
        m.update()
        return {
            'error':error,
            'obj_vals':obj_val,
            'status':status,
            'rt':rt,
            'sol':m
        }

    def complError(self: Self, m: gp.Model) -> float:
        """
        Computes the complementarity error for the solution obtained from Gurobi.

        Args:
            m (gp.Model): The Gurobi model instance with the solution.
        Returns:
            float: The complementarity error.
        """

        x_sol = np.array([m.getVarByName(f"x[{f},{p},{a}]").X for f,p,a in product(self.firms, self.portfolios,self.assets)])
        x_c_sol = np.array([m.getVarByName(f"x_c[{f},{p},{a}]").X for f,p,a in product(self.firms, self.portfolios,self.assets)])
        lmb_sol = np.array([m.getVarByName(f"lmb[{f}]").X for f in self.firms])
        lmb_c_sol = np.array([m.getVarByName(f"lmb_c[{f}]").X for f in self.firms])
        
        error = np.vdot(x_sol,x_c_sol)
        error += np.vdot(lmb_sol,lmb_c_sol)
        return error
    
    def addBilinearDC(
            self: Self, 
            m: gp.Model, 
            x1: gp.Var | list[gp.Var], 
            x2: gp.Var | list[gp.Var], 
            c: gp.Var | list[gp.Var], 
            name: str
        ) -> gp.Model:
        """
        Dynamically adds bilinear DC constraints to the Gurobi model.

        Args:
            m (gp.Model): The Gurobi model instance.
            x1 (gp.Var or list of gp.Var): First variable(s) in the bilinear term.
            x2 (gp.Var or list of gp.Var): Second variable(s) in the bilinear term.
            c (gp.Var or list of gp.Var): Constant term(s) in the bilinear term.
            name (str): Base name for the added variables and constraints.
        Returns:
            gp.Model: The updated Gurobi model with bilinear DC constraints added.
        """

        m.update()
        
        t = m.getVarByName('t')
        if isinstance(x1,gp.Var) and isinstance(x2,gp.Var):
            u = m.addVar(name=f"{name}-u")
            v = m.addVar(lb=-np.inf,name=f"{name}-v")
            uk,vk = 0,0
            dc_1 = m.addConstr(self.alpha*u**2 - self.beta*vk**2 - 2*self.beta*vk*(v-vk) - c <= t)
            dc_2 = m.addConstr(self.beta*v**2 - self.alpha*uk**2 - 2*self.alpha*uk*(u-uk) + c <= t)
            m.addConstr(x1 == self.N_mat[0,0]*u + self.N_mat[0,1]*v)
            m.addConstr(x2 == self.N_mat[1,0]*u + self.N_mat[1,1]*v)
            self.constr_to_update = [*self.constr_to_update, dc_1, dc_2]
            self.vars_to_update.append((u,v,c))
            
        else: 
            n = len(x1)
            u = m.addVars(n,name=f"{name}-u")
            v = m.addVars(n,lb=-np.inf,name=f"{name}-v")
            uk,vk = np.zeros(n), np.zeros(n)
            for i in range(n):
                dc_1 = m.addConstr(self.alpha*u[i]**2 - self.beta*vk[i]**2 - 2*self.beta*vk[i]*(v[i]-vk[i]) - c[i] <= t)
                dc_2 = m.addConstr(self.beta*v[i]**2 - self.alpha*uk[i]**2 - 2*self.alpha*uk[i]*(u[i]-uk[i]) + c[i] <= t)
                m.addConstr(x1[i] == self.N_mat[0,0]*u[i] + self.N_mat[0,1]*v[i])
                m.addConstr(x2[i] == self.N_mat[1,0]*u[i]+ self.N_mat[1,1]*v[i])
                self.constr_to_update = [*self.constr_to_update, dc_1, dc_2]
                self.vars_to_update.append((u[i],v[i],c[i]))

        return m
    
    def updateBilinearDC(
            self: Self,
            m: gp.Model,
            u: gp.Var,
            v: gp.Var,
            uk: float,
            vk: float,
            c: gp.Var
        ):
        """
        Updates the bilinear DC constraints in the Gurobi model based on new variable values.

        Args:
            m (gp.Model): The Gurobi model instance.
            u (gp.Var): Variable u in the bilinear term.
            v (gp.Var): Variable v in the bilinear term.
            uk (float): Previous value of variable u.
            vk (float): Previous value of variable v.
            c (gp.Var): Constant term in the bilinear term.
        Returns:
            gp.Model: The updated Gurobi model with bilinear DC constraints updated.
        """

        
        t = m.getVarByName('t')
        dc_1 = m.addConstr(self.alpha*u**2 - self.beta*vk**2 - 2*self.beta*vk*(v-vk) - c <= t)
        dc_2 = m.addConstr(self.beta*v**2 - self.alpha*uk**2 - 2*self.alpha*uk*(u-uk) + c <= t)
        self.constr_to_update = [*self.constr_to_update, dc_1, dc_2]

        return m



    def solveDCA(
            self: Self,
            rho_init: float,
            delta1: float,
            delta2: float,
            t_eps: float,
            x_eps: float,
            verbosity: int = 1,
            max_iters: int = 500
        ) -> dict[str, Any]:
        """
        Solves the multi-portfolio equilibrium problem using the DCA-BL algorithm.
        Args:
            rho_init (float): Initial penalty parameter for the DCA-BL algorithm.
            delta1 (float): DCA-BL penalty parameter.
            delta2 (float): DCA-BL penalty parameter.
            t_eps (float): DCA-BL feasibility tolerance.
            x_eps (float): DCa-BL convergence tolerance.
            verbosity (int, optional): Level of verbosity for output. Defaults to 1.
            max_iters (int, optional): Maximum number of iterations for the DCA-BL algorithm
        Returns:
            dict: A dictionary containing the solution details:
                - 'error': Complementarity error of the solution.
                - 'obj_vals': Objective values for each firm.
                - 'status': Gurobi solver status code.
                - 'rt': Runtime of the solver in seconds.
                - 'sol': The Gurobi model instance with the solution.

        """
        
        evals,U = sc.linalg.eigh(self.V)
        D = np.diag(np.sqrt(np.real(evals)))
        covSqrt = U@D@U.T

        m = gp.Model()
        m.params.NonConvex = 0
        m.params.QCPDual = 1
        m.params.OutputFlag = 0
        
        
        # Parametrs 
        rho = rho_init
        w_k = {(f,p,a):0 for f,p,a in product(self.firms,self.portfolios,self.assets)}
        

        ##### VARIABLES ####
        ub = np.inf
        # Primary Variables
        x = m.addVars(self.firms,self.portfolios,self.assets,lb=0,ub=ub,name='x')
       

        # Dual Variables 
        lmb = m.addVars(self.firms,lb=0,name='lmb')
        #gma = m.addVars(self.firms,lb=-np.inf,name='gma')
        gma = m.addVars(self.firms,self.portfolios,lb=-np.inf,name='gma')


        # Complementarity Variables
        lmb_c = m.addVars(self.firms,lb=0,name='lmb_c')
        #gma_c = m.addVars(self.firms,lb=0,name='gma_c')
        x_c = m.addVars(self.firms,self.portfolios,self.assets,lb=0,name='x_c')

        # Auxillary Variables
        std = m.addVars(self.firms,lb=0,name='std')
        Vx = m.addVars(self.firms,self.portfolio,self.assets,lb=0)

        z = m.addVars(self.firms,self.portfolios,self.assets,lb=-np.inf,name='z')
        y = m.addVars(self.firms,self.portfolios,self.assets,lb=-np.inf,name='y')
        xi = m.addVars(self.firms,self.portfolios,self.assets,lb=-np.inf,name='xi')
        eta = m.addVars(self.firms,self.portfolios,self.assets,lb=-np.inf,name='eta')

        # DCA Variables 
        t = m.addVar(lb=0,name='t')
        
        ###### CONSTRAINTS ######
        dc_con = {}
        for f in self.firms:
            
            # Linear Constraints 
            m.addConstrs((gp.quicksum([x[f,p,a] for a in self.assets]) == 1 for p in self.portfolios),
                        name='budget')

            
    
            # Defining Constraints
            m.addConstrs(
                        gp.quicksum(
                                    [
                                        covSqrt[self.idx[pp,aa],self.idx[p,a]]*x[f,p,a] 
                                        for p, a in product(self.portfolios, self.assets)
                                    ]
                                    ) == Vx[f,pp,aa] 
                        for pp,aa in product(self.portfolios,self.assets)
                        )
            m.addConstrs((z[f,p,a] == self.grad_f(x,f)[self.idx[p,a]] + gma[f,p] 
                          for p,a in product(self.portfolios, self.assets)),
                          name='z_def')
            
            m.addConstrs((y[f,p,a] == self.mu[p,a]*std[f] 
                                        - self.f_inv[f]*gp.quicksum([self.V[self.idx[p,a],self.idx[pp,aa]]*x[f,pp,aa] 
                                                                     for pp,aa in product(self.portfolios,self.assets)])  
                         for p,a in product(self.portfolios, self.assets)),
                         name='y_def')
            
            
            m.addConstr(gp.quicksum([self.mu[p,a]*x[f,p,a] for p,a in product(self.portfolios,self.assets)]) + self.f_inv[f]*std[f] - self.R_min[f] == lmb_c[f],
                        name='lmb_c_def')
            
            
            # # Quadratic Constraints
            m.addQConstr(gp.quicksum([Vx[f,p,a]*Vx[f,p,a] for p,a in product(self.portfolios, self.assets)])<=std[f]**2, name='var_convex')
        
    
            m.addConstrs((x_c[f,p,a] == xi[f,p,a] - eta[f,p,a] for p,a in product(self.portfolios,self.assets)),
                         name='x_c_def')
        
            # DCA Constraints
            Vwk = [sum(self.V[self.idx[pp,aa],self.idx[p,a]]*w_k[f,p,a] for p,a in product(self.portfolios,self.assets)) for pp,aa in product(self.portfolios,self.assets)]
            h_qk = sum(
                        sum(w_k[f,p,a]*w_k[f,pp,aa]*self.V[self.idx[p,a],self.idx[pp,aa]] 
                            for p,a in product(self.portfolios,self.assets)) 
                        for pp,aa in product(self.portfolios,self.assets)
                        )
            dc_con[f] = m.addQConstr(std[f]**2 - h_qk - gp.quicksum([Vwk[self.idx[p,a]]*(x[f,p,a]-w_k[f,p,a]) for p,a in product(self.portfolios,self.assets)]) <= t, 'DC')
            self.constr_to_update.append(dc_con[f])
        
            # Bilinear DC Constraints 
            m = self.addBilinearDC(m,lmb[f],lmb_c[f],0,f'lmb[{f}]')
            for p,a in product(self.portfolios,self.assets):
                m = self.addBilinearDC(m,std[f],z[f,p,a],xi[f,p,a],f'std_z[{f},{p},{a}]')
                m = self.addBilinearDC(m,lmb[f],y[f,p,a],eta[f,p,a],f'lmb_y[{f},{p},{a}]')
                m = self.addBilinearDC(m,x[f,p,a],x_c[f,p,a],0,f"x[{f},{p},{a}]")
        
        m.setObjective(rho*t,GRB.MINIMIZE)
       
        w_k = np.zeros(len(m.getVars())+2)
        m.update()
        converged = False
        st = time.time()
        for k in range(max_iters):
            
            m.optimize()
            status = m.getAttr("Status")
            
            
            try:
                w_kp1 = np.array([v.X for v in m.getVars()])
               
            except: continue
            
            t_kp1 = t.X
            
            
            total_diff = linalg.norm(w_kp1 - w_k, ord=1)
            try:
                dual_norm = np.abs(sum(m.getAttr(GRB.Attr.Pi,m.getConstrs())))
                dual_norm += np.abs(sum(m.getAttr(GRB.Attr.QCPi,m.getQConstrs())))
            except: 
                dual_norm = 0

            compl_error = self.complError(m)
            if ((total_diff < x_eps) or (t_kp1 < t_eps)): 
                test = True
            if ((total_diff < x_eps) or (t_kp1 < t_eps)):
                converged = True
                if verbosity>0:
                    print(f"{k+1 : <10}{total_diff: ^30}{t_kp1:^30}{rho:^10}")
                
                break
                
            r = min(1/total_diff, dual_norm + delta1) 
            if rho < r:
                rho += delta2
        
            
             
            m.setObjective(rho*t)


            # Update
            w_k= w_kp1
            for f in self.firms:
                Vwk = [sum(self.V[self.idx[pp,aa],self.idx[p,a]]*w_k[self.f_map[f,p,a]] for p,a in product(self.portfolios,self.assets)) for pp,aa in product(self.portfolios,self.assets)]
                h_qk = sum(
                            sum(w_k[self.f_map[f,p,a]]*w_k[self.f_map[f,pp,aa]]*self.V[self.idx[p,a],self.idx[pp,aa]] 
                                for p,a in product(self.portfolios,self.assets)) 
                            for pp,aa in product(self.portfolios,self.assets)
                            )
                m.remove(dc_con[f])
                dc_con[f] = m.addQConstr(std[f]**2 - h_qk - gp.quicksum([Vwk[self.idx[p,a]]*(x[f,p,a]-w_k[self.f_map[f,p,a]]) for p,a in product(self.portfolios,self.assets)]) <= t, 'DC')
            
            # Update Bilinear Terms
            for con in self.constr_to_update:
                m.remove(con)
            self.constr_to_update = []
            for u,v,c in self.vars_to_update:
                uk, vk = u.X, v.X
                m = self.updateBilinearDC(m,u,v,uk,vk,c)
            m.update()
            if verbosity >0:
                print(f"{k+1 : <10}{total_diff: ^30}{t_kp1:^30}{rho:^10}")
        rt = time.time() - st
       
        if converged:
            obj_val = {f:self.f(x,f) for f in self.firms}
            error = self.complError(m)
            sol = {v.VarName:v.X for v in m.getVars()}
        else:
            obj_val = {f:-1 for f in self.firms}
            try:
                error = self.complError(m)
            except:
                error = -1
            
            sol = -1
        if k == 0:
            t_kp1 = -1
        return {
            'rt':rt, 
            'error':error,
            'iters': k+1,
            't_final': t_kp1,
            'sol':sol,
            'converged':converged
        }
