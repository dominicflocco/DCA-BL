from pyomo.environ import * 
from pyomo.mpec import *
import numpy as np 
import pandas as pd
import os,time
from numpy import linalg
from ncp_methods.waterEquilibriumDCA import waterEquilibriumDCA

TAU1_OPT = 0.667
TAU2_OPT = 0.259

def main():
    """
    Conducts domain tightening experiments on the water equilibrium model.
    Saves results to an Excel file.
    """

    datafile = '../data/water_equilibrium/wqual_model_data_new_export.xlsx'

    model = waterEquilibriumDCA()
    model.read_data(datafile)
    results = {}
    beta1s = np.linspace(0.1,1.0,10)
    beta2s = np.linspace(0.1,1.0,10)
    
    for beta1 in beta1s:
        for beta2 in beta2s:
            model = waterEquilibriumDCA()
            model.read_data(datafile)
            print(f" ===== b1={beta1}, b2={beta2} ======")
            res = model.solveDCA(
                rho_init=100,
                delta1=100,
                delta2=100,
                eps=1e-6,
                max_iters=10000,
                beta1=beta1, 
                beta2=beta2,
                verbosity=1
            )
            results[(beta1, beta2)] = [
                res['runtime'], 
                res['iters'], 
                res['error'], 
                res['converged'],
                res['status']
            ]
    
    results_df = pd.DataFrame.from_dict(
        results, 
        columns=['Runtime', 'Iters', 'Error', 'Converged?', 'Status'],
        orient='index'
    )
    results_df.set_index(
        pd.MultiIndex.from_tuples(
            tuples=results.keys(), 
            names=['b1', 'b2']
        ), 
        inplace=True
    )
    with pd.ExcelWriter(f"domain-tightening.xlsx", engine="openpyxl") as writer: 
        results_df.to_excel(writer, merge_cells=False)

if __name__ == '__main__':
    main()