import os,time,openpyxl, pickle,sys,logging
from pyomo.environ import * 
from pyomo.mpec import *
import numpy as np 
import pandas as pd
from src.lcp_methods.LcpDCSolver import LcpDCSolver
from tqdm import tqdm
import scipy as sp
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

def build_lcp_matrix_elastic(
        A: Dict[int, np.ndarray], 
        beta: Dict[int, float], 
        timesteps: List[int], 
        players: List[int]
    ) -> np.ndarray:
    """
    Build the LCP coefficient matrix for the price-maker market equilibrium 
    problem with elastic demand.

    Args:
        A (Dict[int, np.ndarray]): Dictionary of constraint matrices for each player.
        beta (Dict[int, float]): Dictionary of inverse demand function slopes for each timestep.
        timesteps (List[int]): List of timesteps.
        players (List[int]): List of player indices.
    Returns:
        np.ndarray: The constructed LCP coefficient matrix M.
    """
    num_timesteps = len(timesteps)
    num_players = len(players)
    H = np.zeros((num_timesteps, num_timesteps))
    Hp = np.zeros((num_timesteps, num_timesteps))
    for t in timesteps:
        Hp[(t-1), (t-1)] = 2*beta[t]
        H[(t-1), (t-1)] = beta[t]

    n = num_players*(2*num_timesteps+1)
    M = np.zeros((n,n))
    mp_size = 2*num_timesteps + 1
    A_mt = {p:-A[p].T for p in players} 
    for p in players:
        idx = (p-1)*mp_size
        M[idx:idx+num_timesteps,idx:idx+num_timesteps] = Hp
        M[idx+num_timesteps:idx+mp_size, idx:idx+num_timesteps] = A[p]
        M[idx:idx+num_timesteps, idx+num_timesteps:idx+mp_size] = A_mt[p]
        for other_p in players:
            if p != other_p:
                idx2 = (other_p-1)*mp_size
                M[idx:idx+num_timesteps, idx2:idx2+num_timesteps] = H
    return M

def build_bp(total_cap: float, max_rates: Dict[int, float], timesteps: List[int]) -> np.ndarray:
    """
    Builds RHS vector b for each player in the price-maker market equilibrium problem.

    Args:
        total_cap (float): Total capacity for the player.
        max_rates (Dict[int, float]): Dictionary of maximum production rates for each timestep.
        timesteps (List[int]): List of timesteps.
    Returns:
        np.ndarray: The constructed RHS vector b.
    """   
    return np.array([total_cap,*[max_rates[t] for t in timesteps]]) 

def build_cp(gamma: float, alpha: Dict[int, float], timesteps: List[int]) -> np.ndarray:
    """
    Builds objective coefficient vector c for each player in the price-maker 
    market equilibrium problem.

    Args:
        gamma (float): Production cost.
        alpha (Dict[int, float]): Dictionary of inverse demand slopes for each timestep.
        timesteps (List[int]): List of timesteps.
    Returns:
        np.ndarray: The constructed objective coefficient vector c.
    """
    return np.array([gamma - alpha[t] for t in timesteps])

def build_lcp_vector(b: Dict[int, np.ndarray], c: Dict[int, np.ndarray], timesteps: List[int], players: List[int]) -> np.ndarray:
    """
    Build the right-hand side vector overall LCP.
    Args:
        b (Dict[int, np.ndarray]): Dictionary of RHS vectors for each player.
        c (Dict[int, np.ndarray]): Dictionary of objective coefficient vectors for each player.
        timesteps (List[int]): List of timesteps.
        players (List[int]): List of player indices.
    Returns:
        np.ndarray: The constructed LCP right-hand side vector q.
    """
    num_timesteps = len(timesteps)
    num_players = len(players)
    n = num_players*(2*num_timesteps + 1)
    pl_size = 2*num_timesteps+1
    q = np.zeros((n,1))
    for p in players: 
        idx = (p-1)*pl_size
        q[idx:idx+num_timesteps,:] = c[p]
        q[idx+num_timesteps: p*pl_size,:] = b[p]
    return q

def build_Ap(timesteps: List[int]) -> np.ndarray:
    """
    Builds the coefficient matrix for each player in the price-maker market equilibrium problem.

    Args:
        timesteps (List[int]): List of timesteps.
    Returns:
        np.ndarray: The constructed coefficient matrix Ap.
    """
    num_timesteps = len(timesteps)
    Ap = np.zeros((num_timesteps+1, num_timesteps))
    Ap[0,range(num_timesteps)] = -np.ones(num_timesteps)
    for t in timesteps: 
        Ap[t,(t-1)] = -1
    return Ap

def generate_problems(nts: List[int], nps: List[int], trials: int=10) -> None:
    """
    Generates random price-maker market equilibrium problem instances and saves them as pickle files.

    Args:
        nts (List[int]): List of numbers of timesteps to consider.
        nps (List[int]): List of numbers of players to consider.
        trials (int): Number of random instances to generate for each (Np, Nt) pair

    """

    outdir = "../data/market_equilibrium/price-maker"

    for Np in nps: 
        for Nt in nts:
            for t in range(trials):
                data = {}
                data['mat'], data['vec'] = generate_price_maker(Np, Nt)
                fname = os.path.join(outdir,f'{Np}-{Nt}-{t}.p')
                pickle.dump(data,open(fname,'wb'))

def generate_price_maker(Np: int, Nt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single random price-maker market equilibrium problem instance.

    Args:
        Np (int): Number of players.
        Nt (int): Number of timesteps.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The LCP matrix and vector for the problem instance.
    """

    players = range(1, Np+1)
    timesteps = range(1, Nt+1)
    alpha = Np*np.random.rand(Nt,1)
    Aps, cps, bps = {},{}, {}
    for p in players:
        top_row = -np.ones(Nt)
        Aps[p] = np.vstack((top_row, -np.eye(Nt)))
        vol_max = Nt*np.random.rand()
        max_rates = np.random.rand(Nt,1)
        bps[p] = np.vstack((vol_max,max_rates))
        gamma = np.random.rand()
        cps[p] = gamma*np.ones((Nt,1)) - alpha

    beta = dict(zip(timesteps,0.1*np.random.rand(Nt)))
    lcp_matrix = build_lcp_matrix_elastic(Aps, beta, timesteps, players)
    lcp_vector = build_lcp_vector(bps,cps, timesteps, players)
    
    return lcp_matrix, lcp_vector

def read_problems(dir):
    """
    Reads price-maker market equilibrium problem instances from pickle files.

    Args:
        dir (str): Directory containing the pickle files.
    Returns:
        Tuple[Dict[Tuple[int, int, int], np.ndarray], Dict[Tuple[int, int, int], np.ndarray]]: 
            Two dictionaries containing the matrices and vectors from the pickle files.
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

def solve_with_PATH(N: int, M: np.ndarray, q: np.ndarray) -> Tuple[ConcreteModel, Any]:
    """
    Solve an LCP using the PATH solver via Pyomo.

    Args:
        N (int): Size of the LCP.
        M (np.ndarray): Coefficient matrix of the LCP.
        q (np.ndarray): Right-hand side vector of the LCP.
    Returns:
        Tuple[ConcreteModel, SolverResults]: The Pyomo model and the solver results.
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

def solve_all_instances():
    """
    Solve all generated price-maker market equilibrium instances 
    and save results to an Excel file.
    """

    rho_init = 100
    eps = 1e-4
    delta1 = 10
    delta2 = 1000
    solvers = ['mosek']
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

    lcp_mats, lcp_vecs = read_problems("../data/market_equilibrium/price-maker")

    key_map = dict(zip(range(len(lcp_mats.keys())), lcp_mats.keys()))


    for i in tqdm(range(len(key_map.keys()))):
        k = key_map[i]
        Np, Nt = k[0],k[1]
        n = Np*(2*Nt + 1)

        # Solve LCP with PATH
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
            lcpSolver = LcpDCSolver(f"LCP-n={n}", sparse=False)
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

            # Solve LCP with DCA-BL
            m_instance = lcpSolver.initializeLCP(
                x_init=np.zeros(n),
                rho_init=rho_init,
                t_init=1
            )

            m_st = time.time()
            m_sol, m_res, m_iters, m_rt, m_rho, flag= lcpSolver.DCASolveLCP(
                m_instance,
                solver="mosek",
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
    
    # Create results DataFrame and save to Excel
    res_df = pd.DataFrame.from_dict(results, orient='index', columns=cols)
    
    res_df.set_index(
        pd.MultiIndex.from_tuples(
            tuples=results.keys(), 
            names=['Np', 'Nt', 'trial','M']), 
            inplace=True
    )
    with pd.ExcelWriter(f"../results/market_equilibrium/rand_price_maker-solver_test.xlsx", engine="openpyxl") as writer: 
        res_df.to_excel(writer,merge_cells=False)

if __name__ == '__main__':

    generate_problems(
        nts=[2,3,5,10,12,15],
        nps=[2,3,5,10,12,15],
        trials=10
    )
    solve_all_instances()