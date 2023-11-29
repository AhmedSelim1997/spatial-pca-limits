#%%
from analysis_functions import find_critical_point
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
with open(os.path.join(output_path,"SS_1d,K=5.pkl"),"rb") as f:
    full_df = pickle.load(f)
n_list=np.unique(full_df.n)
L_list = np.unique(full_df.L)
m_list = np.unique(full_df.m)
# %%
n=n_list[-5]
L=L_list[-5]
a = full_df[full_df.n == n]
a = a[a.L == L]
x = a.p_1.values
plt.scatter(np.log10(m_list),x)
# inx = np.min(np.where(tau_list>1/np.sqrt(n*L)))
predicted_inx = find_critical_point(x[::-1])
# plt.axvline(x=np.log10(tau_list[inx]), color='red', linestyle='--', label='Vertical Line at x=3')
plt.axvline(x=np.log10(m_list[predicted_inx]), color='blue', linestyle='--', label='Vertical Line at x=3')
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
