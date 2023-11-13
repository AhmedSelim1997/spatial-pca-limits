#%%
import numpy as np
import pickle
import os
import sys
from functools import partial
module_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(module_directory)
from utils import TWp
data_directory = "../../data/biased_samples/d=1/"
K_list = [5,7,9,11]
n_list = np.linspace(2,100,50).astype(int)
L = 5000
#%%
TWp_vectorized = np.vectorize(TWp)
def TWp_array(data,n_list,L):
    result1 = np.zeros_like(data[:,:,:,0])
    result2 = np.zeros_like(data[:,:,:,1])
    for i,n in enumerate(n_list):
        result1[i,:,:] = TWp_vectorized(data[i,:,:,0],n,L,0)
        result2[i,:,:] = TWp_vectorized(data[i,:,:,1],n-1,L,1)
    return np.divide(result2,result1)
# %%
valid = []
threshold = 2
for K in K_list:
    files = os.listdir(data_directory+f"K={K}")
    for file in files:
        try:
            with open(os.path.join(data_directory,f"K={K}",file),"rb") as f:
                data = pickle.load(f)
            result = TWp_array(data,n_list,L)
            indices = np.where(result>threshold)
            if len(indices[0]) !=0:
                valid.append([K,file,indices])
        except:
            print(K,file)
            continue


# %%
