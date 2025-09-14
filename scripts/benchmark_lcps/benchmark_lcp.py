from pyomo.environ import * 
from pyomo.mpec import *
import numpy as np 
import scipy as sp
import pandas as pd
import os,time,sys,h5py
import scipy as sc
from scipy import io
from numpy import linalg
import logging,os
import pickle
import argparse
from typing import Any, Dict, List, Tuple

from src.lcp_methods.LcpDCSolver import LcpDCSolver

dir_path = os.path.dirname(os.path.realpath(__file__))

def read_test_mats(
        pbs: List[int], 
        sizes: List[int]
    ) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], np.ndarray]]:
    """
    Read test matrices and vectors from .mat files. 

    Args:
        pbs (List[int]): List of problem numbers to read.
        sizes (List[int]): List of problem sizes to read.
    Returns:
        Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], np.ndarray]]: 
            Two dictionaries containing the matrices and vectors from the .mat files.
    """
    tempdir = os.path.join(dir_path, 'data', 'benchmark_lcps')
    mats, vecs = {}, {}
    
    for file in os.listdir(tempdir):
        if file.endswith(".mat"):
            split_str = file.split("-")
            pb = int(split_str[0][3:])
            n = int(split_str[1][1:-4])
            
            if n in sizes and pb in pbs:
                print(n, pb)
                file_path = os.path.join(tempdir,file)
                with h5py.File(file_path, 'r') as f:
                    # Load sparse matrix components
                    values = f['M/data'][:]   
                    row_indices = f['M/ir'][:]   
                    col_pointers = f['M/jc'][:] 
                    mats[(f"LCP{pb}",n)] = sc.sparse.csr_matrix((values, row_indices, col_pointers))
                    vecs[(f"LCP{pb}",n)] = f['q'][:].flatten()
    return mats, vecs

def check_compl(z: np.ndarray, M: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the complementarity error for a given solution.
    Args:
        z (np.ndarray): Solution vector.
        M (np.ndarray): Matrix in the LCP.
        q (np.ndarray): Vector in the LCP.
    Returns:
        float: The complementarity error.
    """
    z = z.reshape(-1, 1)
    q = q.reshape(-1, 1)
    return np.vdot(z, M @ z + q)

def dcabl_solve(
        N:int,
        M:np.ndarray, 
        q:np.ndarray, 
        x_init:np.ndarray, 
        rho_init:float, 
        delta1:float, 
        delta2:float, 
        max_iters:int, 
        verbosity:int=1
    ) -> Tuple[int, int, float, float]:
    """
    Solve the LCP using the DCA-BL method.

    Args:
        N (int): Size of the LCP.
        M (np.ndarray): Matrix in the LCP.
        q (np.ndarray): Vector in the LCP.
        x_init (np.ndarray): Initial solution vector.
        rho_init (float): Initial penalty parameter.
        delta1 (float): Parameter for updating rho.
        delta2 (float): Parameter for updating rho.
        max_iters (int): Maximum number of iterations.
        verbosity (int): Level of verbosity for logging.
    Returns:
        Tuple[int, int, float, float]: A tuple containing the status code, number of iterations,
            runtime, and complementarity error.
    """

    N_mat = np.array([[1,1],[1,-1]])
    lcpSolver = LcpDCSolver(f"DCA-BL-n{N}", sparse=True)
    
    model = lcpSolver.constructLCP(
        N=N,
        M1=M, 
        c1=q, 
        N_mat=N_mat,
        alpha=np.ones(N),
        beta=np.ones(N)
    )
    
    m_instance = lcpSolver.initializeLCP(
        x_init=x_init,
        rho_init=rho_init,
        t_init=1
    )
    
    m_sol, m_res, m_iters, m_rt, m_rho, flag= lcpSolver.DCASolveLCP(
        m_instance,
        solver="mosek",
        eps=1e-5,
        delta1=delta1,
        delta2=delta2,
        max_iters=max_iters,
        verbosity=verbosity
    )

    if m_iters == max_iters:
        m_val = np.array([m_sol.x[i].value for i in range(N)])
        error = check_compl(m_val,M,q)
        code = 1
    elif m_res.solver.status == SolverStatus.ok:
        m_val = np.array([m_sol.x[i].value for i in range(N)])
        error = check_compl(m_val,M,q)
        code = 0
    else: 
        m_val = -1
        error = -1
        code = 2
    return code, m_iters, m_rt, error

def solve_test_problems(pbs: List[int], sizes: List[int]) -> None:
    """
    Solve test problems for the DCA-BL method.

    Args:
        pbs (List[int]): List of problem numbers to solve.
        sizes (List[int]): List of problem sizes to solve.
    """
    params = pickle.load(open(os.path.join(dir_path,"DCA_BI_params.p"),"rb"))

    verbosity = 1
    lcp_mats, lcp_vecs = read_test_mats(pbs,sizes)
    cols = [
        "DCA-BL Status", 
        "DCA-BL Iters", 
        "DCA-BL Runtime", 
        "DCA-BL Error"
    ]

    
    key_map = dict(zip(range(len(lcp_mats.keys())), lcp_mats.keys()))
    results = {}
    for n in sizes: 
        for p in pbs:
            k = (f'LCP{p}', n)
            M, q = lcp_mats[k], lcp_vecs[k]
        
            # Solve with DCA-BI
            rho,delta1,delta2 = params.get((1000,p), params[(1000,7)])
            print(f"Solving LCP{p}-n{n} with DCA-BI...")

            
            dca_bi_res = dcabl_solve(
                n,M,q,
                                        x_init=np.zeros(n), 
                                        rho_init=rho,
                                        delta1=delta1,
                                        delta2=delta2,
                                        max_iters=500,
                                        verbosity=verbosity)
            
            print(f"DCA-BL Solve Complete: {dca_bi_res}")
            
            
            results[k] = [*dca_bi_res]
            res_df = pd.DataFrame.from_dict(
                results, 
                orient='index', 
                columns=cols
            )
            res_df.set_index(
                pd.MultiIndex.from_tuples(
                    tuples=results.keys(), 
                    names=['Problem', 'Dim.']
                ), 
                inplace=True
            )
    with pd.ExcelWriter(os.path.join(dir_path, f"benchmarks_new-n={n}-p={p}.xlsx"), engine="openpyxl") as writer: 
        res_df.to_excel(writer,merge_cells=False)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark LCP solvers.")
    parser.add_argument('--sizes', nargs='+', type=int, required=True, help='List of problem sizes')
    parser.add_argument('--pbs', nargs='+', type=int, required=True, help='List of problem numbers')
    parser.add_argument('--ptype', type=str, default='asym', help='Problem type for density mode')
    parser.add_argument('--densities', nargs='+', type=int, help='Densities for density mode')

    args = parser.parse_args()

    sizes = args.sizes
    pbs = args.pbs

    print("Solving for problem sizes ", sizes)
    print("Solving for problems", pbs)

    solve_test_problems(pbs, sizes)

    

