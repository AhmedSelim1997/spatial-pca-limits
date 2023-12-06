#%%
import numpy as np
import pandas as pd
import pickle
import os
from scripts.analysis.analysis_functions_0 import create_full_eigenvalue_df
import matplotlib.pyplot as plt
#%%
if __name__ == "__main__":
    output_path = "../../results/full_data"
    data_path = "../../data/"
    ## Collecting the paths of data from all simulations
    input_paths = [data_path+"pop_split/"] + [data_path + f"SS_1d/{K}/" for K in os.listdir(data_path + "SS_1d")] + [data_path + f"SS_2d/{K}/" for K in os.listdir(data_path + "SS_2d")] + [data_path + f"cont/{N}/" for N in os.listdir(data_path + "cont")]
    for input_path in input_paths:
        if "pop_split" in input_path:
            file_name = "pop_split"
            param = "Fst"
        else:
            file_name = ",".join(input_path.split("/")[-3:-1])
            param = "m"
        df = create_full_eigenvalue_df(input_path,param)
        with open(os.path.join(output_path,f"{file_name}.pkl"),"wb") as f:
            pickle.dump(df,f)

# %%
