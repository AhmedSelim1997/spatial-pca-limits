#%%
import numpy as np
import pandas as pd
import pickle
import os
from analysis_functions import create_full_eigenvalue_df
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

#%%
def find_critical_point(vec, window_size=10):
    ## It smooths and differentiates the the vector, then finds the index i where all derivative values after index i (till max value) are greater than that of index i
    smoothed_vec = np.convolve(vec, np.ones(window_size)/window_size, mode='valid') ## Smoothing convolution
    derivative_vec = np.gradient(smoothed_vec)
    max_inx = np.argmax(derivative_vec)
    for i,val in enumerate(derivative_vec[1:max_inx]):
        sub_derivatives = derivative_vec[i:max_inx]
        if np.all(sub_derivatives>0):
            return i
    return len(vec)-1
#%%
output_path = "../../results/full_data"
with open(os.path.join(output_path,"pop_split.pkl"),"rb") as f:
    full_df = pickle.load(f)
#%%
n_list=np.unique(full_df.n)
L_list = np.unique(full_df.L)
tau_list = np.unique(full_df.Fst)
n=n_list[0]
L=L_list[2]
a = full_df[full_df.n == n]
a = a[a.L == L]
x = a.p_1.values
plt.scatter(np.log10(tau_list),x)
inx = np.min(np.where(tau_list>1/np.sqrt(n*L)))
predicted_inx = find_critical_point(x)
plt.axvline(x=np.log10(tau_list[inx]), color='red', linestyle='--', label='Vertical Line at x=3')
plt.axvline(x=np.log10(tau_list[predicted_inx]), color='blue', linestyle='--', label='Vertical Line at x=3')
# %%
predicted_tau=[]
theoretical_tau=[]
nL_list=[]
n_used = []
L_used = []
for n in n_list:
    for L in L_list:
        a = full_df[full_df.n == n]
        a = a[a.L == L]
        try:
            x = a.p_1.values
            critical_point = tau_list[find_critical_point(x)]
            if critical_point!=tau_list[-1]:
                predicted_tau.append(critical_point)
                theoretical_tau.append(1/np.sqrt(n*L))
                nL_list.append(n*L)
                n_used.append(n)
                L_used.append(L)
        except:
            continue
predicted_tau=np.array(predicted_tau)
theoretical_tau=np.array(theoretical_tau)
# %%
start=0
plt.scatter(nL_list[start:],predicted_tau[start:])
plt.plot(nL_list[start:],theoretical_tau[start:],c="red")
plt.xscale("log")
plt.yscale("log")
# %%

# %%
