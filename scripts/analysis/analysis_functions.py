#%%
import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import savgol_filter
#%%
def create_full_eigenvalue_df(input_path,param):
    l = len(param) + 1
    data_folders = np.array(os.listdir(input_path))
    data_folders = np.array([file for file in data_folders if "eigenvalues.pkl" in os.listdir(input_path + "/" + file)])
    params_list = np.array([file[l:] for file in data_folders]).astype(float)
    order = np.argsort(params_list)
    params_list = params_list[order]
    data_folders = data_folders[order]
    with open(os.path.join(input_path,data_folders[0])+"/eigenvalues.pkl","rb") as f:
        x = pickle.load(f)
        # n_list = np.unique(x.n.values).astype(int)
        # L_list = np.unique(x.L.values).astype(int)
    column_names =[param]+list(x.columns)
    full_df = pd.DataFrame(columns = column_names)
    for i,folder in enumerate(data_folders):
        with open(os.path.join(input_path,folder) + "/eigenvalues.pkl","rb") as f:
            df = pickle.load(f)
        df[param] = [params_list[i]]*len(df.index)
        df = df[[param]+list(df.columns[:-1])]
        full_df = pd.concat([full_df,df])
    return full_df

def find_critical_point(vec):
    ## It smooths and differentiates the the vector, then finds the index i where all derivative values after index i (till max value) are greater than that of index i
    window_size=int(len(vec)/10)
    smoothed_derivatives=savgol_filter(vec,window_length=window_size,polyorder=3,deriv=1)
    cum_ave = np.divide(np.cumsum(vec),np.arange(1,1+len(vec)))
    function_increasing = smoothed_derivatives>0
    more_than_cum_ave = vec>cum_ave
    y = np.logical_or(function_increasing,more_than_cum_ave)
    for i in range(len(vec)):
        if np.all(y[i:]):
            return i
    return len(vec)-1
