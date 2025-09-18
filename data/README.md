# Data Files
All data files for computational experiments are located in this subdirectory. Each problem class has its own subdirectory, which contains data files and a Readme file describing the data file structure for that problem class. The problem classes are as follows:
1. Randomly generated LCPs `random_lcps`
2. Benchmark LCPs from the literature `benchmark_lcps`
3. Market equilibrium LCPs (price-taker and price-maker models) `market_equilibrium`
4. Multi-portfolio equilibrium NCP `portfolio_opt`
5. Water network equilibrium NCP `water_equilibrium`

Due to the size of some data files, not all data files used in the experiments are included in this repository. However, problem generators for all problem classes are included in the `src` subdirectory, and can be used to generate additional problem instances as needed.
