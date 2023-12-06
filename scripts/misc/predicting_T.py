#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.fft import fft

# %%
m=0.01
n=50
K=5
m_test = np.linspace(0.001,0.1,100)
n_test = np.arange(1,101)
K_test = np.arange(20)

m_pred = regr.predict(np.hstack((np.array([K]*100).reshape(-1,1),m_test.reshape(-1,1),np.array([n]*100).reshape(-1,1))))
n_pred = regr.predict(np.hstack((np.array([K]*100).reshape(-1,1),np.array([m]*100).reshape(-1,1),n_test.reshape(-1,1))))
K_pred = regr.predict(np.hstack((K_test.reshape(-1,1),np.array([m]*20).reshape(-1,1),np.array([n]*20).reshape(-1,1))))

# %%
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].plot(K_test,K_pred)
ax[0].set_xlabel("K")
ax[1].plot(m_test,np.log(m_pred))
ax[1].set_xlabel("m")
ax[2].plot(n_test,n_pred)
ax[2].set_xlabel("n")
ax[0].set_ylabel("T")
fig.suptitle("T varying with K,m,n for 1d SS model")

# %%
def theor_eig_1d(K,m,n):
    N = 1000
    M = m * N
    T_list = [(x * (K - x)) / (2 * M * K) for x in np.arange(1, ((K - 1) / 2) + 1)]
    T_list2 = np.array(T_list+T_list[::-1]) 
    T_list_full = [1]+ list(1+T_list2)
    T_tilde = np.unique(np.real(fft(T_list_full)[1:]))
    return (1 / regr.predict(np.array([K,m,n]).reshape(1,-1))) * (1 - n * T_tilde)
def find_cutoff_m(K,n,L):
    half_K = int((K-1)/2)
    gamma=1+np.sqrt(n/L)
    m_list_temp = np.geomspace(1e-6,0.5,40)
    solutions = np.array([theor_eig_1d(K,m,n)-gamma for m in m_list_temp])
    best_indices = [np.where(solutions[:,i]<0)[0][0]-1 for i in range(half_K)]
    cutoff_m = []
    for i in range(half_K):
        best_index = best_indices[i]
        m_list_temp = np.geomspace(m_list_temp[best_index],m_list_temp[best_index+1],25)
        solutions = [np.abs(theor_eig_1d(K,m,n)[i]-gamma) for m in m_list_temp]
        best_index = np.argmin(solutions)
        cutoff_m.append(m_list_temp[best_index])
    return cutoff_m
def find_cutoff_n(K,m,L):
    half_K = int((K-1)/2)
    n_list = np.linspace(1,200,100).astype(int)
    gamma=1+np.sqrt(n_list/L)
    gamma = np.hstack((gamma.reshape(-1,1),gamma.reshape(-1,1)))
    eig_list = np.array([theor_eig_1d(K,m,n) for n in n_list])
    solutions = eig_list-gamma
    best_indices = [np.where(solutions[:,i]<0)[0][-1]-1 for i in range(half_K)]
    return n_list[np.array(best_indices)]
def find_cutoff_L(K,m,n):
    half_K = int((K-1)/2)
    L_list = np.linspace(1e3,1e5,100).astype(int)
    gamma=1+np.sqrt(n/L_list)
    gamma = np.hstack((gamma.reshape(-1,1),gamma.reshape(-1,1)))
    eig_list = np.array([theor_eig_1d(K,m,n) for n in n_list])
    solutions = eig_list-gamma
    best_indices = [np.where(solutions[:,i]<0)[0][-1]-1 for i in range(half_K)]
    return L_list[np.array(best_indices)]
# %%
K=5
n=50
L=5000
# %%
find_cutoff_m(K,n,L)
# %%
cutoffs = np.zeros((len(K_list),len(n_list),len(L_list),5))
for i,K in enumerate(K_list):
    half_K = int((K-1)/2)
    for j,n in tqdm(enumerate(n_list)):
        for k,L in enumerate(L_list):
            cutoffs[i,j,k,:half_K] = find_cutoff_m(K,n,L)
# %%
output_path = "../../results/full_data"
with open(os.path.join(output_path,"new_df.pkl"),"rb") as f:
    full_df = pickle.load(f)
# %%
n_list=np.unique(full_df.n)
L_list = np.unique(full_df.L)
m_list = np.unique(full_df.m)

# %%
n=n_list[-1]
L=L_list[-1]
x = full_df[(full_df.n == n) & (full_df.L == L)].l_1.values
cutoff_m = find_cutoff_m(5,n,L)[0]
predicted_inx = empirical_m_cutoff(x[::-1],n,L)
predicted_cutoff_m = m_list[::-1][predicted_inx]
plt.axvline(x=cutoff_m,c="red")
plt.axvline(x=predicted_cutoff_m,c="blue")
plt.scatter(m_list,x)
plt.xscale("log")
# %%
def empirical_m_cutoff(vec,n,L):
    edge = (1+np.sqrt(n/L))**2
    return np.where(vec>edge)[0][0]
# %%
def preprocess_T(demography,N=1000):
    data_dir = "../../data/" + demography + "/"
    data = np.zeros((1,4))
    folders = os.listdir(data_dir)
    for folder in tqdm(folders):
        K = int(folder[2:])
        data_folders = os.listdir(os.path.join(data_dir,folder))
        for data_folder in data_folders:
            m = float(data_folder[2:])
            if "branch_lengths.pkl" in os.listdir(data_dir+f"K={K}/m={m}"):
                T_list = pd.read_pickle(data_dir+f"K={K}/m={m}/branch_lengths.pkl")/(K*N)
                data_len = len(T_list)
                data_temp = np.hstack((np.array([K]*data_len).reshape(-1,1),np.array([m]*data_len).reshape(-1,1),np.arange(1,data_len+1).reshape(-1,1),T_list.reshape(-1,1)))
                data = np.vstack((data,data_temp))
    return data[1:,:-1],data[1:,-1]
# %%
def train_regressor_T(demography,N=1000):
    X, y = preprocess_T(demography,N)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    regr = RandomForestRegressor(random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print(f"MSE = {mean_squared_error(y_test, y_pred)}")
    return regr
# %%
def theor_eig_1d(regr,K,m,n,N=1000):
    M = m * N
    T_list = [(x * (K - x)) / (2 * M * K) for x in np.arange(1, ((K - 1) / 2) + 1)]
    T_list2 = np.array(T_list+T_list[::-1]) 
    T_list_full = [1]+ list(1+T_list2)
    T_tilde = np.unique(np.real(fft(T_list_full)[1:]))
    return (1 / regr.predict(np.array([K,m,n]).reshape(1,-1))) * (1 - n * T_tilde)
# %%
