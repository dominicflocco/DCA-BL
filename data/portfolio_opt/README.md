# Portfolio Optimization Problem Instances

This directory contains multi-portofolio equilibrium instances of Nonlinear Complementarity Problems (NCPs) derived from a financial market model. The overall NCP instance is composed of the data defining the Karush-Kuhn-Tucker (KKT) conditions of each agent's chance-constrained optimization problem. The mathematical details of the model are described in the [online supplement](<fill-url>). Importantly, the size of the problem dimension is a function of the number of firms, assets and portofolios considered in the model.

Instances are randomized by sampling monthly return data from the S&P 500 from January 2012 to November 2021. The data are available in `Monthly Returns.csv`, which contains the monthly returns of 500 assets over this period. The source file `src/portfolio_opt/multiPortfolioDataReader.py` contains the necessary functions to read the data and generate the instances. 

The instances are stored as `.pkl` files, which can be read using the `pickle` module in Python. Each `.pkl` file contains two dictionary with keys that define the problem instance: 
- `portfolio`: the data defining the financial market
    - Each key is a portfolio ID, and the value is another dictionary with the following keys
        - `assets`: a list of asset IDs
        - `returns`: a 2D array of asset returns for each asset in the portfolio
        - `mu`: a 1D array of expected asset returns
        - `V`: a 2D array of asset return covariances
- `firms`: the data defining each firm
    - Each key is a firm ID, and the value is another dictionary with the following keys:
        - `B`: budget of the firm
        - `R`: minimum return level for the firm
        - `mat`: matrix defining quadratic transaction cost function
        - `F_inv`: inverse of the cumulative distribution function of the risk measure

The naming convention for the files is as follows: `multiportfolio_p{num_portfolios}-a{num_assets}-f{num_firms}-{index}.pkl`. For example, the file `multiportfolio_p10-a10-f2-k0.pkl` contains a portfolio optimization instance with 10 portfolios, 10 assets per portfolio, and 2 firms. The files can be read in Python as follows:

```python
from src.ncp_methods.multiPortfolioDataReader import multiPortfolioDataReader

reader = multiPortfolioDataReader(datafile='data/portfolio_opt/Monthly Returns.csv')
portfolio_data, firm_data = reader.generateInstance(
        numPortfolios=10,
        assetsPerPortfolio=10,
        numFirms=2
)
```

