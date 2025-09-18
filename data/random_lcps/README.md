# Randomized LCP Problem Instances

This directory contains randomly generated Linear Complementarity Problem (LCP) instances, which are defined by a matrix \( M \) and a vector \( q \). Random instances can be generated with varying sizes and coefficient matrix properties, such as density, symmetry and positive definiteness. The density of the matrix can be any float between 0 and 1, where 0 indicates a completely sparse matrix and 1 indicates a completely dense matrix. For symmetry and positive definiteness, the following options are available: symmetric positive definite (PSD), asymmetric indefinite (ASYM), and symmetric indefinite (SID). 

Due to the size of some data files, not all data files used in the experiments only the instances of size $n = 100$ are included in this repository. However, the problem generators used to generate the instances can be found in the `src/problem_generators` subdirectory, and can be used to generate additional problem instances as needed. Instructions for generating additional instances are provided below.

## Data File Structure

Random LCP instances are stored as `.mat` files, which can be read using the `scipy.io.loadmat` function in Python. Each `.mat` file contains two variables: `M`, which is the coefficient matrix, and `q`, which is the right-hand side vector. The naming convention for the files is as follows: `{size}-{density}-{index}.mat`. For example, the file `100-0.5-1.mat` contains a random LCP instance of size 100 with a coefficient matrix density of 0.5, and is the first instance generated with these parameters. Files can be read in Python as follows:

```python
from scipy import io
data = io.loadmat('100-0.5-1.mat')
M = data['M']
q = data['q']
```

## Generating Additional Instances

To generate additional random LCP instances, use the respective MATLAB script located in the `src/problem_generators` subdirectory. The following scripts are available:
- `generate_asym_lcpmat.m`: Generates symmetric positive definite (PSD) instances.
- `generate_psd_lcpmat.m`: Generates asymmetric indefinite (ASYM) instances.
- `generate_sid_lcpmat.m`: Generates symmetric indefinite (SID) instances.

Each script takes the following input parameters:
- `n`: Size of the LCP instance (number of variables).
- `T`: The number of instances to generate for each density level.
- `tempdir`: The directory where the generated `.mat` files will be saved.

By default, the scripts generate `T` instances of each density level in the set {0.1, 0.2, ..., 0.9, 1.0}. To generate instances with different density levels, modify the `densities` variable in the script.