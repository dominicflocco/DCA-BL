from pyomo.environ import * 
from pyomo.mpec import *
import numpy as np 
import pandas as pd
import os,time
from numpy import linalg
import gurobipy as gp
import scipy as sc
import random,pickle
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from src.ncp_methods.multiPortfolioSolver import multiPorfolioSolver
from src.ncp_methods.multiPortfolioSolver_short import ShortMultiPorfolioSolver
from src.ncp_methods.multiPortfolioDataReader import multiPortfolioDataReader

datafile = '../data/portfolio_opt/Monthly Returns.csv'
instpath = '../data/portfolio_opt/problem_instances'
outpath = '../results/portfolio_opt'


def generate_instances(
        datafile: str, 
        dims: List[Tuple[int, int]],
        num_seeds: int=20,
        num_firms: list[int]=[2]
    )-> None:
    """
    Generate random multi-portfolio optimization instances and save them as pickle files.

    Args:
        datafile (str): Path to the CSV file containing stock return data.
        dims (List[Tuple[int, int]]): List of tuples specifying (number of portfolios
            per instance, number of assets per portfolio).
        num_seeds (int): Number of random instances to generate for each dimension.
        num_firms (list[int]): List of numbers of firms to consider.

    """
    reader = multiPortfolioDataReader(datafile)

    for n_f in num_firms:
        for n_p,n_a in dims:
            
            k = 0
            satisfied = False
            while not satisfied:
                print(f" ================= p{n_p}-a{n_a}-f{n_f}-k{k} =================")
                
                # generate random instance
                portfolio, firm = reader.generateInstance(
                    numPortfolios=n_p,
                    assetsPerPortfolio=n_a,
                    numFirms=n_f
                )
                data = {
                    'portfolio':portfolio,
                    'firm':firm
                }


                solver = multiPorfolioSolver(data)

                # try solving instance with Gurobi to ensure feasibility
                noncon_res = solver.solveKKT()

                # if instance is feasible, save it 
                if noncon_res['status'] == 2:
                    pickle.dump(
                        data,
                        open(os.path.join(instpath,f"multiportfolio_p{n_p}-a{n_a}-f{n_f}-k{k}.p"),'wb')
                    )                                
                    if k == num_seeds-1:
                        satisfied=True
                    else:
                        k += 1



def solve_all_instances(dims: List[Tuple[int, int]]= [(2,3),(3,5),(5,5)], n_firms: list[int]=[2]) -> None:
    """
    Solve all generated multi-portfolio optimization instances and save results to an Excel file.

    Args:
        dims (List[Tuple[int, int]]): List of tuples specifying (number of portfolios
            per instance, number of assets per portfolio).
        n_firms (list[int]): List of numbers of firms to consider.
    """

    cols = [
        "Nonconvex RT", 
        'Nonconvex Error',
        'DCA RT', 
        'DCA Error',
        'DCA Iters',
        't_final'
    ]
    
    results = {}

    for p,a in dims: 
        for f in n_firms:
            print(f" ======================= p{p}-a{a}-f{f} =====================")
            for t in tqdm(range(20)):
                
                datafile = os.path.join(instpath,f"multiportfolio_p{p}-a{a}-f{f}-k{t}.p")
                inst = pickle.load(open(datafile,'rb'))

                solver = multiPorfolioSolver(inst)

                noncon_res = solver.solveKKT()


                dca_res = solver.solveDCA(
                    rho_init=100, 
                    delta1=100,
                    delta2=100,
                    t_eps=1e-4,
                    x_eps=1e-4,
                    eps=1e-5,
                    verbosity=0, 
                    max_iters=2500
                )

                
               
                print(f"============== p{p}-a{a}-f{f}-t{t} =============")
                print("\n-----GUROBI NON-CONVEX SOL-------")
                print(f"Error: {noncon_res['error']}")
                print(f"RT: {noncon_res['rt']}")
                
                print("\n-----------DCA SOL-------------")
                print(f"Error: {dca_res['error']}")
                print(f"RT: {dca_res['rt']}")
                
                cur_res = [
                    noncon_res['rt'],
                    noncon_res['error'],
                    dca_res['rt'],
                    dca_res['error'],
                    dca_res['iters'],
                    dca_res['t_final']
                ]

                results[(f,p,a,t)] = cur_res
                print(f"======================================================\n")

            res_df = pd.DataFrame.from_dict(results, orient='index', columns=cols)
            res_df.set_index(pd.MultiIndex.from_tuples(tuples=results.keys(), names=['Firms','Portfolios','Assets','Instance']), inplace=True)
            with pd.ExcelWriter(os.path.join(outpath,f"multiportfolio_results.xlsx"), engine="openpyxl") as writer: 
                res_df.to_excel(writer,merge_cells=False)



if __name__ == '__main__':

    dims = [(2,3),(3,5),(5,5)]
    n_firms = [2]
    generate_instances(datafile,dims,num_seeds=20,num_firms=n_firms)
    solve_all_instances(dims, n_firms)
