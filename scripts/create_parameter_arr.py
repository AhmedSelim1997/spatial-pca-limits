## This is a general script that outputs a .csv file that can be read when submitting an array job in SLURM

import numpy as np
import pandas as pd
from itertools import product

# param_names = [
#     "Fst"
# ]

# param_values = {
#     param_names[0]: np.round(np.geomspace(1e-5,1e-1,200),8)
# }

# param_csv = pd.DataFrame(list(product(*[param_values[param_name] for param_name in param_names])),columns=param_names)

# filename = "pop_split.csv"
# param_csv.to_csv("../docs/parameter_csv_files/"+filename)

param_names = [
    "N",
    "m"
]

param_values = {
    # param_names[0]: [5,7,9,11],
    # param_names[1]: np.round(np.geomspace(1e-4,0.5,50),5),
    param_names[0]: [500,1000,2000],
    param_names[1]: np.round(np.geomspace(1e-2,1e-1,25),5)[1:-1]
}

param_csv = pd.DataFrame(list(product(*[param_values[param_name] for param_name in param_names])),columns=param_names)

filename = "cont_params_more.csv"
param_csv.to_csv("../docs/parameter_csv_files/"+filename)