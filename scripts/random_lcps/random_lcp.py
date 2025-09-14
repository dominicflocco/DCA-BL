from pyomo.environ import * 
from pyomo.mpec import *
import numpy as np 
import pandas as pd
import os,time,sys
from scipy import io
from numpy import linalg
from tqdm import tqdm
import logging, os
from typing import Any, Dict, List, Tuple
import argparse

from src.lcp_methods.LcpDCSolver import LcpDCSolver

dir_path = os.path.dirname(os.path.realpath(__file__))
out_path = os.path.join(dir_path, '../results/random_LCP')


def read_mat(
        dir: str, 
        densities: List[int]
    ) -> Tuple[Dict[Tuple[int, int, int], np.ndarray], Dict[Tuple[int, int, int], np.ndarray]]:
    """
    Read .mat files from the specified directory and filter them based on the given densities.
    Args:
        dir (str): Directory containing the .mat files.
        densities (List[int]): List of densities to filter the files.
    Returns:
        Tuple[Dict[Tuple[int, int, int], np.ndarray], Dict[Tuple[int, int, int], np.ndarray]]: 
            Two dictionaries containing the matrices and vectors from the .mat files.
    """

    mats, vecs = {}, {}
    for file in os.listdir(dir):
        if file.endswith(".mat"):
            split_str = file.split("-")
            n = int(split_str[0])
            d = int(split_str[1])
            t = int(split_str[2][:-4])
            
            if d in densities:
                mats[(n,d,t)] = io.loadmat(os.path.join(dir,file))['M']
                vecs[(n,d,t)] = io.loadmat(os.path.join(dir,file))['q']
    return mats, vecs


def solve_random_instances(n: int, densities: List[float], solver: str, pb_type='psd') -> None:
    """
    Solve random instances of LCP using the specified subproblem solver.

    Args:
        n (int): Size of the LCP.
        densities (List[float]): List of densities for the random instances.
        solver (str): Subproblem solver to use ('gurobi', 'mosek', etc.).
        pb_type (str): Type of problem ('psd', 'is', 'asym').

    """

    rho_init = 100
    eps = 1e-4
    delta1 = 100
    delta2 = n
    cols = [
        f"{solver} Status", 
        f"{solver} Condition", 
        f"{solver} Warning Flag",
        f"{solver} RT", 
        f"{solver} Iters", 
        f"{solver} Error"
    ]
    results = {}

    d_name = densities[-1]

    # Read LCP matrices and vectors
    lcp_mats, lcp_vecs = read_mat(
        os.path.join(dir_path,f'n{n}-{pb_type}'),
        densities
    )
    key_map = dict(zip(range(len(lcp_mats.keys())), lcp_mats.keys()))
    
    for i in tqdm(range(len(key_map.keys()))):

        k = key_map[i]

        # Initialize DCA-BL LCP Solver
        lcpSolver = LcpDCSolver(f"LCP-n={n}", sparse=True)
        
        # Construct Random LCP
        model = lcpSolver.constructLCP(
            N=n,
            M1=lcp_mats[k], 
            c1=lcp_vecs[k], 
            N_mat=np.array([[1,1],[1,-1]]),
            alpha=np.ones(n),
            beta=np.ones(n)
        )
        
        # Solve Random LCP with DCA-BL
        instance = lcpSolver.initializeLCP(
            x_init=np.zeros(n),
            rho_init=rho_init,
            t_init=1
        )

        # Solve Random LCP with DCA-BL
        sol, res, iters, rt, rho, flag, error = lcpSolver.DCASolveLCP(
            instance,
            solver=solver,
            eps=eps,
            delta1=delta1,
            delta2=delta2,
            max_iters=500,
            verbosity=1
        )
        
        if res is not None:
            status = str(res.solver.status)
            condition = str(res.solver.termination_condition)
        else: 
            status = "terminated"
            condition = 'unsolvable'
        
        # Store Solve Results
        results[k] = [status, condition, flag, rt, iters, error]
    
    # Save Results to Excel
    res_df = pd.DataFrame.from_dict(results, orient='index', columns=cols)
    
    res_df.set_index(
        pd.MultiIndex.from_tuples(
            tuples=results.keys(), 
            names=['size (n)', 'density', 'trial']
        ), 
        inplace=True
    )
    excel_writer = pd.ExcelWriter(
        os.path.join(out_path,f"rand_LCP-{solver}-n={n}-{d_name}-{pb_type}.xlsx"), 
        engine="openpyxl"
    )

    with excel_writer as writer:
        res_df.to_excel(writer,merge_cells=False)

def main():
    """
    Main function to parse arguments and solve random LCP instances.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--densities", type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10], help="List of densities for random instances")
    parser.add_argument("--sizes", type=int, nargs='+', default=[100], help="List of sizes for random instances")
    parser.add_argument("--pb_type", type=str, default='psd', help="Type of problem: 'psd', 'is', 'asym'")
    args = parser.parse_args()
    ds = args.densities
    ns = args.sizes
    pb_type = args.pb_type

    for n in ns: 
        solve_random_instances(n, ds, pb_type=pb_type)

if __name__ == "__main__":
    main()

    
    