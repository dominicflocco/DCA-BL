# Market Equilibrium Problem Instances

This directory contains market equilibrium instances of Linear Complementarity Problems (LCPs) derived from both price-taker and price-maker models. The overall LCP instance is defined by a matrix $M$ and a vector $q$, which are composed of the data defining the Karush-Kuhn-Tucker (KKT) conditions of each agent. The mathematical details of the model are described in the [online supplement](<fill-url>). Importantly, the size of the problem dimension is a function of the number of agents and time periods considered in the model. 

Instances are stored as `.pkl` files, which can be read using the `pickle` module in Python. Each `.pkl` file contains a dictionary with keys: `mat`, which is the coefficient matrix, and `vec`, which is the right-hand side vector. The naming convention for the files is as follows: `{num_agents}-{num_periods}-{index}.pkl`. For example, the file `10-5-1.pkl` contains a price-maker market equilibrium instance with 10 agents and 5 time periods. The files can be read in Python as follows:

```python
import pickle
data = pickle.load(open('10-5-1.pkl', 'rb'))
M = data['mat']
q = data['vec']
```

## Generating Additional Instances
Problem instances are generated in the script `scripts/market_equilibrium/price_maker_LCP.py` for price-maker models and `scripts/market_equilibrium/price_taker_LCP.py` for price-taker models, by the functions `generate_price_maker` and `generate_price_taker`, respectively. The following input parameters can be modified in the script to generate additional instances:

```python
# Example of generating 10 instances with 10 agents and 5 time periods
num_agents = 10
num_periods = 5
generate_problems(nts=num_agents, ntp=num_periods, trials=10) # generate 10 instances
matrices, vectors = read_problems(dir='data/market_equilibrium/price-maker') # read generated instances
```
