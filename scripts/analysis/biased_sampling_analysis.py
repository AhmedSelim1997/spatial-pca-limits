#%%
import numpy as np
import pickle
import os
import sys
import pandas as pd
from tqdm import tqdm
module_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(module_directory)
from utils import TWp
data_directory = "../../data/biased_samples/d=1/"
K_list = [5,7,9,11]
n_list = np.linspace(2,100,50).astype(int)
x_list = np.geomspace(0.01,1,20)
a_list = a_list = np.arange(1,4)
L = 5000
#%%
TWp_vectorized = np.vectorize(TWp)
def TWp_array(data,n_list,L):
    ## The ratio between the p-value of the first and second eigenvalues for a single simulation
    result1 = np.zeros_like(data[:,:,:,0])
    result2 = np.zeros_like(data[:,:,:,1])
    for i,n in enumerate(n_list):
        result1[i,:,:] = TWp_vectorized(data[i,:,:,0],n,L,0)
        result2[i,:,:] = TWp_vectorized(data[i,:,:,1],n-1,L,1)
    return np.divide(result2,result1)
def indices_to_values(lists,index_arr):
    values = np.zeros_like(index_arr).astype(float)
    for i,List in enumerate(lists):
        values[:,i] = np.array([List[j] for j in index_arr[:,i]])
    return values
# %%
params_of_interest = []
threshold = 10
for K in K_list:
    files = os.listdir(data_directory+f"K={K}")
    for file in files:
        try:
            with open(os.path.join(data_directory,f"K={K}",file),"rb") as f:
                data = pickle.load(f)
            result = TWp_array(data,n_list,L)
            indices = np.where(result>threshold)
            if len(indices[0]) !=0:
                params_of_interest.append([K,file,indices])
        except:
            print(K,file)
            continue
#%%
num_params = [len(params[2][0]) for params in params_of_interest]
params_array = np.zeros((1,5))
for i,num in enumerate(num_params):
    temp_arr = np.zeros((num,5))
    temp_arr[:,0] = params_of_interest[i][0]
    temp_arr[:,1] = float(params_of_interest[i][1][2:])
    temp_arr[:,2:] = indices_to_values([n_list,x_list,a_list],np.array(params_of_interest[i][2]).T)
    params_array = np.vstack((params_array,temp_arr))
params_array = params_array[1:,:]
params_df = pd.DataFrame(params_array,columns = ['K','m','n','x','a'])
params_df.set_index(['K', 'm','a'], inplace=True)
# # %%
# with open("../../results/full_data/biased_sampling_params.pkl","wb") as f:
#     pickle.dump(params_df,f)
# #%%
# with open("../../results/full_data/biased_sampling_params.pkl","rb") as f:
#     params_df = pickle.load(f)
# %%
max_x_arr = np.zeros((1,5))
max_n_arr = np.zeros((1,5))
for index in tqdm(params_df.index.unique(),desc="processing"):
    rows = params_df.loc[index]
    max_n = rows.groupby("x").max()
    max_x = rows.groupby("n").max()
    max_x_arr = np.vstack((max_x_arr,np.hstack((np.tile(index,(len(max_x),1)),max_x.index.values.reshape(-1,1),max_x.values.reshape(-1,1)))))
    max_n_arr = np.vstack((max_n_arr,np.hstack((np.tile(index,(len(max_n),1)),max_n.index.values.reshape(-1,1),max_n.values.reshape(-1,1)))))

max_x_arr = max_x_arr[1:,:]
max_n_arr = max_n_arr[1:,:]
max_x_pd = pd.DataFrame(max_x_arr,columns = ['K','m','a','n','max_x'])
max_x_pd.set_index(['K', 'm','a'], inplace=True)

max_n_pd = pd.DataFrame(max_n_arr,columns = ['K','m','a','x','max_n'])
max_n_pd.set_index(['K','m','a'], inplace=True)

#%%
with open("../../results/full_data/biased_max_n.pkl","wb") as f:
    pickle.dump(max_n_pd,f)
with open("../../results/full_data/biased_max_x.pkl","wb") as f:
    pickle.dump(max_x_pd,f)
# %%
