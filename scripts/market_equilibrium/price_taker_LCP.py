import os,time,openpyxl, pickle,sys,logging
from pyomo.environ import * 
from pyomo.mpec import *
import numpy as np 
import scipy as sp
import pandas as pd
from src.lcp_methods.LcpDCSolver import LcpDCSolver
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

def np_to_pyo(arr: np.ndarray) -> dict[Any, Any]:
    """
    Convert a NumPy array to a Pyomo-compatible dictionary format.
    Args:
        arr (np.ndarray): The NumPy array to convert.
    Returns:
        dict: A dictionary representation of the array suitable for Pyomo.
    """
    if len(arr.shape) == 1:
        return {i: arr[i] for i in range(arr.shape[0])}
    else:
        return {(i,j): arr[i,j] for i in range(arr.shape[0]) for j in range(arr.shape[1])}
def build_Mp(Q: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Build the augmented LCP matrix Mp for one player from the quadratic cost matrix Q 
    and constraint matrix A.

    Args:
        Q (np.ndarray): The quadratic cost matrix.
        A (np.ndarray): The constraint matrix.
    Returns:
        np.ndarray: The augmented LCP matrix Mp.
    """
    if Q.shape[0] != Q.shape[1]:
        print("Warning. Q is not square")
        return None
    n = Q.shape[0]
    Mp = np.zeros((n+A.shape[0], n+A.shape[0]))
    for r in range(n):
        for c in range(n):
            Mp[r,c] = Q[r,c]

    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            Mp[r+n, c] = A[r,c]
    A_mt = -A.T 

    for r in range(A_mt.shape[0]):
        for c in range(A_mt.shape[1]):
            Mp[r, c+n] = A_mt[r,c]
    
    return Mp
def build_lcp_matrix_elastic(
        Q: Dict[int, np.ndarray], 
        A: Dict[int, np.ndarray], 
        beta: Dict[int, float], 
        timesteps: List[int], 
        players: List[int]
    ) -> np.ndarray:
    """
    Build the LCP matrix for the price-taker elastic demand model.

    Args:
        Q (Dict[int, np.ndarray]): Dictionary of quadratic cost matrices for each player.
        A (Dict[int, np.ndarray]): Dictionary of constraint matrices for each player.
        beta (Dict[int, float]): Dictionary of price sensitivity parameters for each timestep.
        timesteps (List[int]): List of timesteps.
        players (List[int]): List of players.
    Returns:
        np.ndarray: The LCP matrix.
    """
    num_timesteps = len(timesteps)
    num_players = len(players)
    n = num_players*(2*num_timesteps+1) + num_timesteps
    num_constr = 2*num_timesteps + 1
    Mps = {p:build_Mp(Q[p], A[p]) for p in players}
    lcp_mat = np.zeros((n, n))
    for p in players:
        idx = (p-1)*(num_constr)
        lcp_mat[idx:num_constr+idx, idx:num_constr+idx] = Mps[p]
        lcp_mat[n-num_timesteps:n, idx:idx+num_timesteps] = np.eye(num_timesteps)
        lcp_mat[idx:idx+num_timesteps,n-num_timesteps:n] = -np.eye(num_timesteps)
    lcp_mat[n-num_timesteps:n,n-num_timesteps:n] = np.diag([1/beta[t] for t in timesteps])
    return lcp_mat

def build_lcp_vector_elastic(
        b: Dict[int, np.ndarray], 
        c: Dict[int, np.ndarray], 
        alpha: Dict[int, float], 
        beta: Dict[int, float], 
        timesteps: List[int], 
        players: List[int]
    ) -> np.ndarray:
    """
    Build the LCP vector for the price-taker elastic demand model.

    Args:
        b (Dict[int, np.ndarray]): Dictionary of right-hand side vectors for each player.
        c (Dict[int, np.ndarray]): Dictionary of linear cost vectors for each player.
        alpha (Dict[int, float]): Dictionary of intercepts of inverse demand curve for each timestep.
        beta (Dict[int, float]): Dictionary of slope of inverse demand curve for each timestep.
        timesteps (List[int]): List of timesteps.
        players (List[int]): List of players.
    Returns:
        np.ndarray: The LCP vector.
    """
    num_timesteps = len(timesteps)
    num_players = len(players)
    n = num_players*(2*num_timesteps+1) + num_timesteps
    pl_dim = 2*num_timesteps + 1
    q = np.zeros((n,1))
    for p in players:
        idx = (p-1)*pl_dim
        q[idx:idx+num_timesteps,:] = c[p]
        q[idx+num_timesteps:idx+2*num_timesteps+1,:] = b[p]
    for t in timesteps:
        q[(n - num_timesteps)+(t-1),:] = -alpha[t]/beta[t]
    return q

def build_Ap(timesteps: List[int]) -> np.ndarray:
    """
    Build the coefficient matrix for player <p> in the price-taker elastic demand model.

    Args:
        timesteps (List[int]): List of timesteps.
    Returns:
        np.ndarray: The coefficient matrix Ap.
    """
    num_timesteps = len(timesteps)
    Ap = np.zeros((num_timesteps+1, num_timesteps))
    Ap[0,range(num_timesteps)] = -np.ones(num_timesteps)
    for t in timesteps: 
        Ap[t,(t-1)] = -1
    return Ap

def generate_price_taker(Np: int, Nt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random price-taker market equilibrium instance.

    Args:
        Np (int): Number of players.
        Nt (int): Number of timesteps.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The LCP matrix and vector.
    """

    players = range(1, Np+1)
    timesteps = range(1, Nt+1)

    Qps, Aps, cps, bps = {}, {}, {}, {}
    for p in players:
        temp = np.random.rand(Nt,Nt)
        Qps[p] = np.dot(temp,temp.transpose())
    
        top_row = -np.ones(Nt)
        Aps[p] = np.vstack((top_row, -np.eye(Nt)))
        cps[p] = np.random.rand(Nt,1)

        vol_max = Nt*np.random.rand()
        max_rates = np.random.rand(Nt,1)
        bps[p] = np.vstack((vol_max,max_rates))

    beta = dict(zip(timesteps,0.1*np.random.rand(Nt)))
    alpha = dict(zip(timesteps,Np*np.random.rand(Nt)))


    lcp_matrix = build_lcp_matrix_elastic(Qps, Aps, beta, timesteps, players)
    lcp_vector = build_lcp_vector_elastic(bps,cps, alpha, beta, timesteps, players)
    
    return lcp_matrix, lcp_vector


def solve_with_PATH(N: int, M: np.ndarray, q: np.ndarray) -> Tuple[ConcreteModel, Any]:
    """
    Solve the LCP using PATH via Pyomo.

    Args:
        N (int): Size of the LCP.
        M (np.ndarray): The LCP matrix.
        q (np.ndarray): The LCP vector.
    Returns:
        Tuple[ConcreteModel, SolverResults]: The Pyomo model and solver results.
    """

    mcp = ConcreteModel()

    mcp.i = RangeSet(0,N-1)
    
    mcp.M = Param(mcp.i, mcp.i, initialize=np_to_pyo(M))
    mcp.q = Param(mcp.i, initialize=dict(zip(range(N), q)))
    mcp.x = Var(mcp.i,initialize=dict(zip(range(N),np.ones(N))),within=NonNegativeReals)

    def mcp_con(m,i):
        return complements(0<= sum(m.M[i,j]*m.x[j] for j in m.i) + m.q[i], m.x[i]>=0)
    mcp.constraint = Complementarity(mcp.i, rule=mcp_con)

    opt = SolverFactory('pathampl')
    try:
        res = opt.solve(mcp,tee=True)
    except: 
        res = None
    return mcp, res

def generate_problems(nts: List[int], nps: List[int], trials: int) -> None:
    """
    Generates and saves random price-taker market equilibrium instances to as pickle files.
    """
    outdir = "../data/market_equilibrium/price-taker"

    for Np in nps: 
        for Nt in nts:
            for t in range(trials):
                data = {}
                data['mat'], data['vec'] = generate_price_taker(Np, Nt)
                fname = os.path.join(outdir,f'{Np}-{Nt}-{t}.p')
                pickle.dump(data,open(fname,'wb'))

def read_problems(dir: str) -> Tuple[Dict[Tuple[int,int,int], np.ndarray], Dict[Tuple[int,int,int], np.ndarray]]:
    """
    Read problem instances from pickle files.

    Args:
        dir (str): Directory containing the pickle files.
    Returns:
        Tuple[Dict[Tuple[int,int,int], np.ndarray], Dict[Tuple[int,int,int], np.ndarray]]: 
            The LCP matrices and vectors.
    """
    mats, vecs = {},{}
    for file in os.listdir(dir):
        if file.endswith(".p"):
            split_str = file.split("-")
            Np = int(split_str[0])
            Nt = int(split_str[1])
            t = int(split_str[2][:-2])
            file = os.path.join(dir,f'{Np}-{Nt}-{t}.p')
            data = pickle.load(open(file, 'rb'))
            mats[(Np,Nt,t)] = data['mat']
            vecs[(Np,Nt,t)] = data['vec'].flatten()
    return mats,vecs

def solve_all_instances():
    """
    Solve all generated price-taker market equilibrium instances with PATH and DCA-BL,
    and save results to an Excel file.
    """
    rho_init = 100
    eps = 1e-4
    delta1 = 10
    delta2 = 1000
    solvers = ['mosek', 'gurobi', 'cplex']
    cols = [
        "PATH Status",
        "PATH Condition",
        "PATH RT"
    ]
    
    for solver in solvers:
        cols += [
            f"{solver} Status", 
            f"{solver} Condition", 
            f"{solver}Warning Flag",
            f"{solver} RT", 
            f"{solver} Iters", 
            f"{solver} penalty",
            f"{solver} Sol. Diff."
        ]
    
               
    results = {}

    lcp_mats, lcp_vecs = read_problems("../data/market_equilibrium/price-taker")

    key_map = dict(zip(range(len(lcp_mats.keys())), lcp_mats.keys()))
    
    for i in tqdm(range(len(key_map.keys()))):
        k = key_map[i]
        Np, Nt = k[0],k[1]
        n = Np*(2*Nt + 1) + Nt
        p_st = time.time()
        p_sol, p_res = solve_with_PATH(n,lcp_mats[k], lcp_vecs[k])
        p_rt = time.time() - p_st
        if p_res is not None:
            p_status = str(p_res.solver.status)
            p_condition = str(p_res.solver.termination_condition)
        else: 
            p_status = 'terminated'
            p_condition = 'unsolvable'
        idx = (k[0],k[1],k[2],n)
        results[idx] = [p_status, p_condition, p_rt]
        for solver in solvers:

            lcpSolver = LcpDCSolver(f"LCP-n={n}, solver={solver}", sparse=False)
            p_val = [p_sol.x[i].value for i in range(n)]
            model = lcpSolver.constructLCP(
                N=n,
                M1=lcp_mats[k], 
                c1=lcp_vecs[k], 
                N_mat=np.array([[1,1],[1,-1]]),
                alpha=np.ones(n),
                beta=np.ones(n),
                obj_rule='penalty'
            )

            #Solve LCP with DCA using Mosek Solver
            m_instance = lcpSolver.initializeLCP(
                x_init=np.zeros(n),
                rho_init=rho_init,
                t_init=1
            )

            m_st = time.time()
            m_sol, m_res, m_iters, m_rt, m_rho, flag = lcpSolver.DCASolveLCP(
                m_instance,
                solver=solver,
                eps=eps,
                delta1=delta1,
                delta2=delta2,
                max_iters=500,
                verbosity=1
            )
            m_rt = time.time() - m_st
            m_val = [m_sol.x[i].value for i in range(n)]
            if m_res is not None:
                m_status = str(m_res.solver.status)
                m_condition = str(m_res.solver.termination_condition)
                m_diff = sum(p_sol.x[i].value - m_sol.x[i].value for i in range(n))
            else: 
                m_status = "terminated"
                m_condition = 'unsolvable'
                m_diff = -1
            
            results[idx].extend(
                [m_status, m_condition, flag, m_rt, m_iters, m_rho, m_diff]
            )

        res_df = pd.DataFrame.from_dict(results, orient='index', columns=cols)
    
        res_df.set_index(pd.MultiIndex.from_tuples(tuples=results.keys(), names=['Np', 'Nt', 'trial','M']), inplace=True)
        with pd.ExcelWriter(f"../results/market_equilibrium/rand_price_taker-solver_test.xlsx", engine="openpyxl") as writer: 
            res_df.to_excel(writer,merge_cells=False)

if __name__ == '__main__':
    
    generate_problems(
        nts=[2,3,5,10,12,15],
        nps=[2,3,5,10,12,15],
        trials=10
    )
    solve_all_instances()