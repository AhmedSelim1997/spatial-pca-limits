## This is a general script that outputs a .csv file that can be read when submitting an array job in SLURM
#%%
import numpy as np
import pandas as pd
from itertools import product

def create_parameter_array(param_names,param_values,filename):
    assert len(param_names) == len(param_values)
    param_values = {param_names[i]:param_values[i] for i in range(len(param_names))}
    param_csv = pd.DataFrame(list(product(*[param_values[param_name] for param_name in param_names])),columns=param_names)
    param_csv.to_csv("../docs/parameter_csv_files/"+filename)


pop_split_param_names = ["Fst"]
pop_split_param_values = [np.round(np.geomspace(1e-5,0.5,200),6)]
pop_split_filename = "pop_split_params.csv"

SS_param_names = ["K","m"]
SS_1d_min=5e-5
SS_1d_mid_1 = 1e-3
SS_1d_mid_2 = 1e-2
SS_1d_max=0.25

SS_2d_min=1e-3
SS_2d_mid = 10**-1.5
SS_2d_max=0.5

SS_1d_m_list = np.round(np.hstack([np.geomspace(SS_1d_min,SS_1d_mid_1,20),np.linspace(SS_1d_mid_1,SS_1d_mid_2,80)[1:],np.geomspace(SS_1d_mid_2,SS_1d_max,20)[1:]]),7)
# SS_2d_m_list = np.round(np.concatenate(np.geomspace(SS_2d_min,SS_2d_mid,50),np.linspace(SS_2d_mid,SS_2d_max,50)),6)


SS_1d_param_values = [[5,7,9,11],SS_1d_m_list]
SS_2d_param_values = [[5,7,9],np.round((SS_2d_max+SS_2d_min)-np.linspace(SS_2d_min,SS_2d_max,num=100),6)[::-1]]

SS_1d_filename = "SS_1d_params.csv"
SS_2d_filename = "SS_2d_params.csv"

cont_param_names = ["N","m"]
cont_param_values = [[500,1000,2000], np.round(np.geomspace(1e-2,0.25,50),5)]
cont_filename = "cont_params.csv"





create_parameter_array(pop_split_param_names,pop_split_param_values,pop_split_filename)
create_parameter_array(SS_param_names,SS_1d_param_values,SS_1d_filename)
create_parameter_array(SS_param_names,SS_2d_param_values,SS_2d_filename)
create_parameter_array(cont_param_names,cont_param_values,cont_filename)
# %%
