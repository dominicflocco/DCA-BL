# Benchmark LCP Instances from the Literature

This directory contains benchmark instances of Linear Complementarity Problems (LCPs) taken from the literature. Each instance is defined by a matrix \( M \) and a vector \( q \), and is stored in a separate `.mat` file. Each of the instances can be generated for any specified size \(n \). The structure of the matrix \( M\) and the vector \(q\) are described in the [online supplement](<fill-url>) to the paper.

Due to the size of some data files, not all data files used in the experiments only the instances of size $n = 100$ are included in this repository. However, the problem generator used to generate the instances can be found in the `src/problem_generators` subdirectory, and can be used to generate additional problem instances as needed. Instructions for generating additional instances are provided below.


## Data Structure

Benchmark LCP instances are stored as `.mat` files, which can be read using the `scipy.io.loadmat` function in Python. Each `.mat` file contains two variables: `M`, which is the coefficient matrix, and `q`, which is the right-hand side vector. The naming convention for the files is as follows: `{instance}-{size}.mat`. For example, the file `lcp6-100.mat` contains the LCP6 test instance of size 100. Files can be read in Python as follows:

```python
from scipy import io
data = io.loadmat('lcp6-100.mat')
M = data['M']
q = data['q']
```

## Generating Additional Instances
To generate additional benchmark LCP instances, use the MATLAB script `benchmark_lcps.m` located in the `src/problem_generators` subdirectory. The script takes the following input parameters:
- `sizes`: A vector of sizes for which to generate the instances (e.g., `[50, 100, 200]`).
- `pb_num`: The problem number corresponding to the desired benchmark instance (e.g., `6` for LCP6, `7` for LCP7, etc.).
- `tempdir`: The directory where the generated `.mat` files will be saved.