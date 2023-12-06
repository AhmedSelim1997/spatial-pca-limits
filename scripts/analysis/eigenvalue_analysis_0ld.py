#%%
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from scripts.analysis.analysis_functions_0 import find_critical_point
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve
#%%
output_path = "../../results/full_data"
with open(os.path.join(output_path,"pop_split.pkl"),"rb") as f:
    full_df = pickle.load(f)
n_list=np.unique(full_df.n)
L_list = np.unique(full_df.L)
tau_list = np.unique(full_df.Fst)
#%%
n=n_list[-4]
L=L_list[-1]
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
start=0
x_values = np.log10(nL_list)
y_values = np.log10(predicted_tau)

coefficients = np.polyfit(x_values, y_values, 1)
line_of_best_fit = np.polyval(coefficients, x_values)

plt.scatter(np.log10(nL_list[start:]),np.log10(predicted_tau[start:]))
plt.plot(np.log10(nL_list[start:]),np.log10(theoretical_tau[start:]),c="red")
plt.plot(x_values, line_of_best_fit, color='yellow', label='Line of best fit')
plt.plot()
# %%

# %%
output_path = "../../results/full_data"
with open(os.path.join(output_path,"new_df.pkl"),"rb") as f:
    full_df = pickle.load(f)
n_list=np.unique(full_df.n)
L_list = np.unique(full_df.L)
m_list = np.unique(full_df.m)
# %%
K=5
n=n_list[10]
L=L_list[15]
a = full_df[full_df.n == n]
a = a[a.L == L]
x = a.p_1.values
m_cutoff = find_cutoff_m(K,n,L)[0]
plt.scatter(m_list,x,marker='.')
plt.axvline(x=m_cutoff, color='red', linestyle='--')
# inx = np.min(np.where(tau_list>1/np.sqrt(n*L)))
# predicted_inx = find_critical_point(x[::-1])
# plt.axvline(x=np.log10(tau_list[inx]), color='red', linestyle='--', label='Vertical Line at x=3')
# plt.axvline(x=m_list[predicted_inx], color='blue', linestyle='--', label='Vertical Line at x=3')
plt.xscale("log")
# %%
n=n_list[5]
L=L_list[5]
a = full_df[full_df.n == n]
a = a[a.L == L]
x = a.p_1.values
plt.scatter(range(len(x)),x[::-1])
plt.axvline(x=predicted_inx, color='red', linestyle='--')
# %%
def theor_eig_1d(n,m,p):
    K = 5
    N = 1000
    M = m * N
    t_ave = 1 + (((K**2)-1) / (12 * M * K))
    T_list = np.array([(x * (K - x)) / (2 * M * K) for x in np.arange(1, ((K - 1) / 2) + 1)])
    cos_transform_list = np.array([np.cos((2 * np.pi * p * x) / K) for x in np.arange(1, ((K - 1) / 2) + 1)])
    T_tilde = 2*np.sum(np.multiply(T_list, cos_transform_list))
    return (1 / t_ave) * (1 - n * T_tilde)

def find_cutoff_m(n,L,p):
    l=1+np.sqrt(n/L)
    m_list_temp = np.geomspace(0.01,0.5,20)
    solutions = np.array([theor_eig_1d(n,L,m,p)-l for m in m_list_temp])
    best_index = np.where(solutions)[0][0]-1
    m_list_temp = np.geomspace(m_list_temp[best_index],m_list_temp[best_index+1],25)
    solutions = [np.abs(theor_eig_1d(n,L,m,p)-l) for m in m_list_temp]
    best_index = np.argmin(solutions)
    return m_list_temp[best_index]

# %%
from itertools import product
cutoff_m_df = pd.DataFrame(np.array(list(product(n_list,L_list,[0]))),columns = ["n","L","cutoff_m"])
cutoff_n_df = pd.DataFrame(np.array(list(product(m_list,L_list,[0]))),columns = ["m","L","cutoff_n"])
cutoff_L_df = pd.DataFrame(np.array(list(product(n_list,m_list,[0]))),columns = ["n","m","cutoff_L"])

# %%
for i,row in cutoff_m_df.iterrows():
    cutoff_m_df.iloc[i,-1] = find_cutoff_m(n=row.n,L=row.L,p=1)

# %%
initial_guess = 10
for i,row in cutoff_n_df.iterrows():
    theor_eig_1d_n = lambda n:  theor_eig_1d(n=n,L=row.L,m=row.m,p=1) 
    cutoff_n_df.iloc[i,-1] = fsolve(theor_eig_1d_n, initial_guess)
# %%
initial_guess = 500
for i,row in cutoff_L_df.iterrows():
    theor_eig_1d_L = lambda L:  theor_eig_1d(L=L,n=row.n,m=row.m,p=1) 
    cutoff_L_df.iloc[i,-1] = fsolve(theor_eig_1d_L, initial_guess)
# %%
n=n_list[-3]
L=L_list[-6]
a = full_df[(full_df.n == n) & (full_df.L == L)]
x = a.p_1.values
theoretical_m = cutoff_m_df[(cutoff_m_df.n==n) & (cutoff_m_df.L==L)].cutoff_m.values[0]
predicted_inx = find_critical_point(x[::-1])
plt.scatter(-np.log10(m_list),x)
plt.axvline(x=-np.log10(theoretical_m), color='red', linestyle='-', label='Vertical Line at x=3')
plt.axvline(x=-np.log10(m_list[::-1][17]), color='blue', linestyle='--', label='Vertical Line at x=3')
# %%
theor_eig1 = []
emp_eig1 = []
n=80
full_df_slice = full_df[(full_df.n==n) & (full_df.L == 4000) & (full_df.p_1>3)]
m_list_smaller = full_df_slice.m.values
for m in m_list_smaller:
    emp_eig1.append(full_df_slice[full_df_slice.m==m].l_1.values[0])
    theor_eig1.append(theor_eig_1d(n,m,1))
plt.scatter(theor_eig1,emp_eig1)
plt.plot(theor_eig1,theor_eig1,c="red")
# %%
theor_eig2 = []
emp_eig2 = []
n=100
full_df_slice = full_df[(full_df.n==n) & (full_df.L == 4000) & (np.isin(full_df.m,m_list_smaller))]
for m in m_list_smaller:
    emp_eig2.append(full_df_slice[full_df_slice.m==m].l_1.values[0])
    theor_eig2.append(theor_eig_1d(n,m,1))
plt.scatter(theor_eig2,emp_eig2)
plt.plot(theor_eig2,theor_eig2,c="red")
# %%
theor_eig1 = np.array(theor_eig1)
theor_eig2 = np.array(theor_eig2)
emp_eig1 = np.array(emp_eig1)
emp_eig2 = np.array(theor_eig2)

# %%
theor_eig_ratio = np.divide(theor_eig1,theor_eig2)
emp_eig_ratio = np.divide(emp_eig1,emp_eig2)
# %%
plt.scatter(theor_eig_ratio,emp_eig_ratio)
plt.plot(theor_eig_ratio,theor_eig_ratio,c="red")







# %%
from scipy.fft import fft
from tqdm import tqdm
from itertools import product
def theor_eig_1d(K,m,data_dir,N=1000,max_n=200):
    M = m * N
    T_list = [(x * (K - x)) / (2 * M * K) for x in np.arange(1, ((K - 1) / 2) + 1)]
    T_list2 = np.array(T_list+T_list[::-1]) 
    T_list_full = [1]+ list(1+T_list2)
    T_tilde = np.unique(np.real(fft(T_list_full)[1:]))
    T = pd.read_pickle(data_dir + f"K={K}/m={m}/branch_lengths.pkl")/(N*K)
    return  np.array([(1 /T[n]) * (1 - (n+1) * T_tilde) for n in range(max_n)])

K_list=[5,7,9,11]
data_dir = "../../data/SS_1d/"
full_arr=np.full((len(K_list),len(m_list),len(n_list),len(L_list),max(K_list)),np.nan)
for i,K in tqdm(enumerate(K_list)):
        half_K = int((K-1)/2)
        for j,m in enumerate(m_list):
            try:
                eig_all_n = theor_eig_1d(K,m,data_dir)
                eig_all_n=eig_all_n[n_list.astype(int)-1,:]
                gamma_list = np.array([1+np.sqrt(n/L) for n,L in product(n_list,L_list)]).reshape(len(n_list),len(L_list))
                result = eig_all_n[:, np.newaxis, :] - gamma_list[:, :, np.newaxis]
                full_arr[i,j,:,:,:half_K] = result
            except: continue
# %%



#%%
data_dir = "../../data/SS_1d/"
K_list = [5,7,9,11]
folders = {K: os.listdir(data_dir+f"K={K}") for K in K_list}
m_list = {}
m_list = np.sort([float(folder[2:]) for folder in folders[5]])
N=1000
# %%
tot_num_T = len(K_list)*len(m_list)*len(n_list)
data = np.zeros((1,4))
for i,K in enumerate(K_list):
    for j,m in enumerate(m_list):
        try:
            T_list = pd.read_pickle(data_dir+f"K={K}/m={m}/branch_lengths.pkl")/(K*N)
            data_temp = np.hstack((np.array([K]*len(n_list)).reshape(-1,1),np.array([m]*len(n_list)).reshape(-1,1),np.linspace(2,100,50).reshape(-1,1),T_list[2:101:2].reshape(-1,1)))
            data = np.vstack((data,data_temp))
        except:
            continue
# %%
X=data[1:,:-1]
y=data[1:,-1]
# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
# %%
K=np.array([5]*100).reshape(-1,1)
# m=np.array([0.01]*100).reshape(-1,1)
m = np.linspace(0.001,0.1,100).reshape(-1,1)
n=np.array([100]*100).reshape(-1,1)
# n=np.arange(1,101).reshape(-1,1)
a = np.hstack((K,m,n))
# %%
plt.plot(m,reg.predict(a))
plt.xscale("log")

# %%
# %%
