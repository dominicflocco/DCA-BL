# Water Equilibrium Problem Instance

The Excel file in this directory contains an instance of a water network equilibrium problem from Boyd et al. 2024. The details of this model are described in the [online supplement](<fill-url>). The script `src/ncp_methods/waterEquilibriumDCA.py` contains the necessary functions to read the data and can be run in Python as follows:

```python
from src.ncp_methods.waterEquilibriumDCA import waterEquilibriumDCA

model = waterEquilibriumDCA().read_data('data/water_equilibrium/water_network_data.xlsx')
```
