[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Heuristic for Complementarity Problems Using Difference of Convex Functions

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[A Heuristic for Complementarity Problems Using Difference of Convex Functions](https://doi.org/10.1287/ijoc.2019.0000) by S. Gabriel, D. Flocco, T. Boomsma, M. Schmidt and M. Lejeune. 
The snapshot is based on 
[this SHA](https://github.com/tkralphs/JoCTemplate/commit/f7f30c63adbcb0811e5a133e1def696b74f3ba15) 
in the development repository. 

**Important: This code is being developed on an on-going basis at 
https://github.com/tkralphs/JoCTemplate. Please go there if you would like to
get a more recent version or would like support**

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2019.0000

https://doi.org/10.1287/ijoc.2019.0000.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{DCABL,
  author =        {Gabriel, Steven and Flocco, Dominic and Boomsma, Trine and Schmidt, Martin and Lejuene, Miguel},
  publisher =     {INFORMS Journal on Computing},
  title =         {A Heuristic for Complementarity Problems Using Difference of Convex Functions},
  year =          {2025},
  doi =           {10.1287/ijoc.2019.0000.cd},
  url =           {https://github.com/INFORMSJoC/2019.0000},
  note =          {Available for download at https://github.com/INFORMSJoC/2019.0000},
}  
```

## Description

The repository contains the implementation of a heuristic for solving linear and nonlinear complementarity problems (LCPs and NCPs) using a difference of convex functions (DC) approach. The method is referred to as DCA-BL (Difference of Convex Function Algorithms for Bilinear Terms). All algorithms are implemented in Python. LCP problem instances are implemented using the Pyomo modeling language, while NCP instances are implemented using gurobipy. The heuristic is based on the DCA framework, which iteratively solves a sequence of convex approximations to the original non-convex problem.

The heuristic is applied to five classes of problems:
1. Randomly generated LCPs `random_lcps`
2. Benchmark LCPs from the literature `benchmark_lcps`
3. Market equilibrium LCPs (price-taker and price-maker models) `market_equilibrium`
4. Multi-portfolio equilibrium NCP `portfolio_opt`
5. Water network equilibrium NCP `water_equilibrium`

Problem generators, data files and results for all problem classes are included in the repository, along with scripts to run the DCA-BL algorithm on each problem class. Each problem class has its own subdirectory under `scripts`, `data` and `results`. Problem generators and the implementation of the DCA-BL algorithm are located in the `src` subdirectory.

See the Readme file in the `data` subdirectory for details on the data file structures for each problem class.


## Results


## Replicating


## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site]().

## Support

For support in using this software, submit an
[issue]().
