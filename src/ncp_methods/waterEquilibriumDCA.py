from pyomo.environ import * 
from pyomo.mpec import *
import numpy as np 
import math
import pandas as pd
import os,time,pickle
from numpy import linalg
import gurobipy as gp
from gurobipy import GRB
import scipy as sc	
from typing import Any, Dict, Self


class waterEquilibriumDCA:
    """
    Class to model and solve a water equilibrium problem using DCA-BL.
    """
    def __init__(self: Self):
        """
        Initializes the waterEquilibriumDCA class.
        """
        self.name = 'Water Equilibrium Problem (DCA)'
        self.constr_to_update = []
        self.vars_to_update = []
        self.N_mat=np.array([[1,1],[1,-1]])
        self.alpha=1
        self.beta=1

    
    def read_data(self: Self, data_file: str) -> None:
        """
        Reads data from an Excel file to set up the water equilibrium problem.
        Args:
            data_file (str): Path to the Excel file containing the data.
        """
        # Load sets
        self.players = pd.read_excel(data_file, sheet_name='P',index_col=0).index.tolist()
        self.constituents = pd.read_excel(data_file, sheet_name='C',index_col=0).index.tolist()
        self.technologies = pd.read_excel(data_file, sheet_name='T',index_col=0).index.tolist()
        self.watersheds = pd.read_excel(data_file, sheet_name='W',index_col=0).index.tolist()
        self.land_uses = pd.read_excel(data_file, sheet_name='L',index_col=0).index.tolist()
        # Read data
        self.s_bar = pd.read_excel(data_file, sheet_name='s_bar', index_col=[0,1,2,3,4]).to_dict()['value']
        self.o_init_bar = pd.read_excel(data_file,sheet_name='l_init_bar', index_col=[0,1]).to_dict()['value']
        self.v_s2 = pd.read_excel(data_file,sheet_name='v_I2', index_col=[0,1,2,3,4,5,6,7]).to_dict()['value']
        self.v_so = pd.read_excel(data_file,sheet_name='v_IY', index_col=[0,1,2,3,4]).to_dict()['value']
        self.v_o2 = pd.read_excel(data_file,sheet_name='v_Y2', index_col=[0,1]).to_dict()['value']
        self.o_allw = pd.read_excel(data_file,sheet_name='l_allw', index_col=[0,1]).to_dict()['value']
        self.u = pd.read_excel(data_file,sheet_name='u', index_col=[0,1,2,3]).to_dict()['value']
        self.f_inv = pd.read_excel(data_file,sheet_name='f_inv', index_col=[0,1]).to_dict()['value']

        self.idx = self.cartesian_prod(self.technologies,self.watersheds,self.land_uses)
        self.V_s2 = {(p,c):np.zeros((len(self.idx),len(self.idx))) for p in self.players for c in self.constituents}
        self.V_so = {(p,c):np.zeros((len(self.idx),1)) for p in self.players for c in self.constituents}
        self.V_sqrt = {(p,c):np.zeros((len(self.idx),len(self.idx))) for p in self.players for c in self.constituents}
        self.i_map = {self.idx[i]:i for i in range(len(self.idx))}
        sym_test = {}
        sqrt_test = {}
        eig_test = {}
        self.V = {}
        mdict = {}
        cond_numbers = {}
        for p in self.players:
            for c in self.constituents:
                for j in self.idx:
                    if (p,c,*j) not in self.s_bar.keys():
                        self.s_bar[(p,c,*j)]= 0
                    if (p,c,*j) in self.v_so.keys():
                        self.V_so[(p,c)][self.i_map[j],0] = self.v_so[p,c,*j]
                    else: 
                        self.v_so[p,c,*j] = 0
                    for k in self.idx:
                        if (p,c,*j,*k) in self.v_s2.keys():
                            self.V_s2[(p,c)][self.i_map[j],self.i_map[k]] = self.v_s2[p,c,*j,*k]
                        else: 
                            self.v_s2[p,c,*j,*k] = 0
                        
                self.V[(p,c)] = np.block([[self.V_s2[(p,c)], self.V_so[(p,c)]],
                                           [self.V_so[(p,c)].T, np.array([[self.v_o2[p,c]]])]])
                
                mdict[f"V_{p}_{c}"] = self.V[(p,c)]
                mdict[f"V_s2_{p}_{c}"] = self.V_s2[(p,c)]
                sym_test[(p,c)] = (self.V[(p,c)] == self.V[(p,c)].T).all()
                evals ,U = sc.linalg.eigh(self.V[(p,c)])
                evals[np.abs(evals)<1e-6] = 0
                
                D = np.diag(np.sqrt(np.real(evals)))
                self.V_sqrt[(p,c)] = U@D@U.T
                
                cond_numbers[(p,c)]= np.linalg.cond(self.V[(p,c)])
                
                sqrt_test[p,c]= np.allclose(self.V_sqrt[(p,c)]@self.V_sqrt[(p,c)], self.V[(p,c)])
                
                
            for j in self.idx:
                if (p,*j) not in self.u.keys():
                    self.u[p,*j] = 0
        
        self.i_map[('aux',)] = -1

    

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
            n = len(x1.values())
            u = m.addVars(x1.keys(),name=f"{name}-u")
            v = m.addVars(x1.keys(),lb=-np.inf,name=f"{name}-v")
            uk,vk = dict(zip(x1.keys(),np.zeros(n))), dict(zip(x1.keys(),np.zeros(n)))
            for i,var in x1.items():
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
            self:Self, 
            rho_init: float, 
            delta1: float, 
            delta2: float, 
            eps: float,
            max_iters: int,
            beta1: float,
            beta2: float,
            inst_num: str = 'NA',
            verbosity: int = 1
        ) -> Dict[str, Any]:
        """
        Formulates and solves the water equilibrium problem using the DCA-BL algorithm and gurobipy.

        Args:
            rho_init (float): Initial penalty parameter for the DCA-BL algorithm.
            delta1 (float): Parameter to adjust rho based on dual norm.
            delta2 (float): Parameter to increase rho if needed.
            eps (float): Convergence tolerance for the DCA algorithm.
            max_iters (int): Maximum number of iterations for the DCA algorithm.
            beta1 (float): Scaling parameter for player 1's upper bound.
            beta2 (float): Scaling parameter for player 2's upper bound.
            inst_num (str, optional): Instance number for logging purposes. Defaults to 'NA'.
            verbosity (int, optional): Level of verbosity for output. Defaults to 1.
        """

        m = gp.Model()
        m.params.NonConvex = 0
        m.params.QCPDual = 1
        m.params.OutputFlag = 0

        i_map = self.i_map
        self.constituents = ['Q']
        self.idx = self.cartesian_prod(self.technologies,self.watersheds,self.land_uses)

        # Parametrs 
        rho = rho_init
        I_k = {p: np.zeros(len(self.idx)+1) for p in self.players}
        

        ######## VARIABLES #######
        
        self.I_idx = self.idx.copy()
        self.I_idx.append(('aux',))
        # Primal Vars
        I = m.addVars(self.players, self.I_idx,name='I')
        
        for p in self.players:
            I[p,'aux'].lb = -1
            I[p,'aux'].ub = -1
        K = m.addVars(self.players, self.constituents,lb=-np.inf,name='K')

        # Dual Vars
        gma_min = m.addVars(self.players, self.constituents,name='gma_min')
        gma_up = m.addVars(self.players, self.idx,name='gma_up')
        pi = m.addVars(self.constituents, lb=-np.inf,name='pi')

        # Auxillary Variables 
        std = m.addVars(self.players,self.constituents,name='std')
        V_s2_I = m.addVars(self.players, self.constituents,self.I_idx,lb=-np.inf,name='V_s2_I')
        
        # Complementarity Variables
        I_c = m.addVars(self.players, self.idx)
        gma_min_c = m.addVars(self.players,self.constituents,name='gma_min_c')
        gma_up_c = m.addVars(self.players, self.idx,name='gma_up_c')

        # DCA Variables 
        t = m.addVar(lb=0,name='t')
        xi = m.addVars(self.players,self.idx,lb=0,name='xi')
        y = m.addVars(self.players,self.idx,lb=0,name='y')

        eta = m.addVars(self.players,self.idx,lb=-np.inf,name='eta')
        z = m.addVars(self.players,self.idx,lb=-np.inf,name='xi_min')


        ########## CONSTRAINTS #########
        # Linear Constraints

        m.addConstrs((pi[c] - gma_min[p,c] == 0 
                      for c in self.constituents 
                      for p in self.players), name='KKT_STAT_K')

    
        m.addConstrs((gp.quicksum([K[p,c] for p in self.players]) == 0 
                      for c in self.constituents),name='KKT_SYS_MC')

        
        m.addConstrs((gp.quicksum([self.V_sqrt[(p,c)][i_map[j],i_map[k]]*I[p,*k] for k in self.I_idx]) == V_s2_I[p,c,*j] 
                      for p in self.players 
                      for c in self.constituents 
                      for j in self.I_idx),name='v_temp_def')
        
        numVars = m.getAttr('NumVars')
        
        p1_up_derived = sum(self.u['p1',*j] for j in self.idx)
        p2_up_derived = sum(self.u['p2',*j] for j in self.idx)
        p1_up = p1_up_derived*beta1
        p2_up = p2_up_derived*beta2
        m.addConstr(gp.quicksum([I['p1',*j] for j in self.idx]) <= p1_up)
        m.addConstr(gp.quicksum([I['p2',*j] for j in self.idx]) <= p2_up)
        
        # Quadratic Constraints 
        quad_expr = {(p,c):gp.quicksum([V_s2_I[p,c,*j]*V_s2_I[p,c,*j] for j in self.I_idx]) 
                     for p in self.players 
                     for c in self.constituents}
        
        m.addConstrs((quad_expr[p,c]  <= std[p,c]**2 
                      for p in self.players 
                      for c in self.constituents), name='SOCC')


        V_Ik = {(p,c):[gp.quicksum([
                            self.V[(p,c)][i_map[i],i_map[j]]*I_k[p][i_map[j]] for j in self.I_idx
                            ]) 
                            for i in self.I_idx] 
                      for p in self.players for c in self.constituents}
        h_Ik = {(p,c):sum(sum(I_k[p][i_map[i]]*I_k[p][i_map[j]]*self.V[(p,c)][i_map[i],i_map[j]] 
                              for i in self.I_idx) 
                              for j in self.I_idx) 
                              for p in self.players 
                              for c in self.constituents}
        dc_con = m.addConstrs((std[p,c]**2 - h_Ik[p,c] 
                               - gp.quicksum([V_Ik[p,c][i_map[i]]*(I[p,*i]-I_k[p][i_map[i]]) for i in self.I_idx]) <= t 
                               for p in self.players 
                               for c in self.constituents), 'DC')
        self.constr_to_update.append(dc_con)

        # Defining Constraints

        # VERSION 1
        m.addConstrs((y[p,*j] == 1 + gma_up[p,*j] 
                      for p in self.players 
                      for j in self.idx),name='y_def')
        
        
        temp_expr = {(p,i): self.f_inv[p,'Q']*gp.quicksum([self.V[p,'Q'][i_map[i],i_map[j]]*I[p,*j] for j in self.I_idx]) 
                     for p in self.players 
                     for i in self.idx}
        m.addConstrs((z[p,*j] == self.s_bar[p,'Q',*j]*std[p,'Q'] + temp_expr[p,j] 
                      for p in self.players 
                      for j in self.idx),name='z_def')
        
        m.addConstrs((I_c[p,*j] == xi[p,*j] - eta[p,*j] 
                      for p in self.players 
                      for j in self.idx),name='I_c_def')
        
        m.addConstrs((gma_min_c[p,c] == gp.quicksum([self.s_bar[p,c,*j]*I[p,*j] for j in self.idx]) 
                      - self.o_init_bar[p,c] + self.f_inv[p,c]*std[p,c] + K[p,c] + self.o_allw[p,c] 
                      for p in self.players 
                      for c in self.constituents),name='gma_min_c_def')
        
        m.addConstrs((gma_up_c[p,*j] == self.u[p,*j] - I[p,*j] 
                      for j in self.idx 
                      for p in self.players),name='gma_up_c_def')


        # # # Bilinear DCA Constraints
        for p in self.players:
            for c in self.constituents:
                for j in self.idx:
                    m = self.addBilinearDC(m,std[p,c],y[p,*j],xi[p,*j],name='std_y')
                    m = self.addBilinearDC(m,gma_min[p,c],z[p,*j],eta[p,*j],name='gma_min_z')
                    m = self.addBilinearDC(m,I_c[p,*j], I[p,*j],0,name='I_compl')
                m = self.addBilinearDC(m,gma_min_c[p,c], gma_min[p,c],0,name='gma_min_compl')
            for j in self.idx:
                m = self.addBilinearDC(m,gma_up_c[p,*j], gma_up[p,*j],0,name='gma_up_compl')
        
        
        
        
        m.setObjective(rho*t,GRB.MINIMIZE)

        u_k = np.zeros(len(self.vars_to_update))
        v_k = np.zeros(len(self.vars_to_update))
        m.update()
        m.write("test.lp")
        add_con = False
        I_k = np.ones(len(self.idx)*len(self.players))
        
        num_vars = m.getAttr('NumVars')
        lin_cons = m.getConstrs()
        q_cons = m.getQConstrs()
        st = time.time()
        converged = False
        iter_log = []
        for k in range(max_iters):
            
            m.optimize()
            status = m.getAttr("Status")
            if status not in [2,13]:
                t_kp1 = -1
                break
            
            I_kp1 = np.array([I[p,*j].X for j in self.idx for p in self.players])
            


            u_kp1 = np.array([var[0].X for var in self.vars_to_update])
            v_kp1 = np.array([var[1].X for var in self.vars_to_update])

            t_kp1 = t.X
        
            total_diff = linalg.norm(I_kp1 - I_k, ord=1)
            total_diff += linalg.norm(u_kp1 - u_k, ord=1)
            total_diff += linalg.norm(v_kp1 - v_k, ord=1)

            try:
                dual_norm = np.abs(sum(m.getAttr(GRB.Attr.Pi,m.getConstrs())))
                
            except:
                dual_norm = 0
            if (total_diff < eps) or (t_kp1 < eps):
                add_con = True
                if verbosity>0:
                    print(f"{k+1 : <10}{total_diff: ^30}{t_kp1:^30}{rho:^10}")
                converged = True
            
                iter_log.append([k+1,total_diff,t_kp1])
                print("\n--------DCA SOLUTION FOUND-------")
                break

            r = min(1/total_diff, dual_norm + delta1) 
            if rho < r:
                rho += delta2
            m.setObjective(rho*t)

            I_k,u_k,v_k = I_kp1,u_kp1,v_kp1

            I_kd  = {p:np.array([I[p,*j].X for j in self.I_idx])for p in self.players}
            V_Ik = {(p,c):[gp.quicksum([
                            self.V[(p,c)][i_map[i],i_map[j]]*I_kd[p][i_map[j]] for j in self.I_idx
                            ]) 
                            for i in self.I_idx] 
                      for p in self.players for c in self.constituents}
            h_Ik = {(p,c):sum(sum(I_kd[p][i_map[i]]*I_kd[p][i_map[j]]*self.V[(p,c)][i_map[i],i_map[j]] for i in self.I_idx) 
                              for j in self.I_idx) 
                              for p in self.players 
                              for c in self.constituents}
            m.remove(dc_con)
            dc_con = m.addConstrs((std[p,c]**2 - h_Ik[p,c] 
                                   - gp.quicksum([V_Ik[p,c][i_map[i]]*(I[p,*i]-I_kd[p][i_map[i]]) for i in self.I_idx]) <= t 
                                   for p in self.players 
                                   for c in self.constituents), 'DC')
            
            # Update Bilinear Terms
            for con in self.constr_to_update:
                m.remove(con)
            self.constr_to_update = []
            for u,v,c in self.vars_to_update:
                uk, vk = u.X, v.X
                m = self.updateBilinearDC(m,u,v,uk,vk,c)
                

            m.update()
            iter_log.append([k+1,total_diff,t_kp1])
            if verbosity >0:
                print(f"{k+1 : <10}{total_diff: ^30}{t_kp1:^30}{rho:^10}")
        if inst_num is not None:
            pickle.dump({'iter_log':iter_log,'p1_up':p1_up,'p2_up':p2_up},
                        open(f'iter_log-{inst_num}.p','wb'))
        rt = time.time() - st
        print(f"Runtime: {rt}")
        print(f"Error: {t_kp1}")
        print(f"Iters: {k}")
        return {
            'runtime':rt, 
            'error': t_kp1, 
            'iters':k-1, 
            'converged':converged,
            'status':status
}

    def cartesian_prod(self: Self, s1: list, s2: list, s3: list) -> list:
        """
        Generates the Cartesian product of three input lists.

        Args:
            s1 (list): First list.
            s2 (list): Second list.
            s3 (list): Third list.
        Returns:
            list: A list of tuples representing the Cartesian product of the input lists.
        """
        prod = []
        for i in s1: 
            for j in s2:
                for k in s3:
                    prod.append((i,j,k))
        return prod
    def np_to_pyo(self: Self, arr: np.ndarray) -> dict:
        """
        Converts a numpy array to a Pyomo Param dictionary format.

        Args:
            arr (np.ndarray): The numpy array to convert.
        Returns:
            dict: A dictionary suitable for initializing a Pyomo Param.
        """

        if len(arr.shape) == 1:
            return {i+1: arr[i] for i in range(arr.shape[0])}
        else:
            return{(i,j): arr[i,j] for i in range(arr.shape[0]) for j in range(arr.shape[1])}