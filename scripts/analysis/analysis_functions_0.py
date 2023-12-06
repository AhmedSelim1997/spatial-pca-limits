#%%
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.fft import fft
import re
#%%
# def find_critical_point(vec):
#     ## It smooths and differentiates the the vector, then finds the index i where all derivative values after index i (till max value) are greater than that of index i
#     window_size=int(len(vec)/10)
#     smoothed_derivatives=savgol_filter(vec,window_length=window_size,polyorder=3,deriv=1)
#     cum_ave = np.divide(np.cumsum(vec),np.arange(1,1+len(vec)))
#     function_increasing = smoothed_derivatives>0
#     more_than_cum_ave = vec>cum_ave
#     y = np.logical_or(function_increasing,more_than_cum_ave)
#     for i in range(len(vec)):
#         if np.all(y[i:]):
#             return i
#     return len(vec)-1


def construct_partial_eigenvalue_df(input_path,param):
    data_folders = np.array(os.listdir(input_path))
    data_folders = np.array([file for file in data_folders if "eigenvalues.pkl" in os.listdir(input_path + "/" + file)]) ## only include folders with eigenvalue output
    params_list = np.array([file.split("=")[1] for file in data_folders]).astype(float) #list of m_values or Fst_values
    order = np.argsort(params_list)
    params_list = params_list[order]
    data_folders = data_folders[order]
    with open(os.path.join(input_path,data_folders[0])+"/eigenvalues.pkl","rb") as f:
        x = pickle.load(f)
    column_names =[param]+list(x.columns)
    full_df = pd.DataFrame(columns = column_names)
    for i,folder in enumerate(data_folders):
        with open(os.path.join(input_path,folder) + "/eigenvalues.pkl","rb") as f:
            df = pickle.load(f)
        df[param] = [params_list[i]]*len(df.index)
        df = df[[param]+list(df.columns[:-1])]
        full_df = pd.concat([full_df,df])
    return full_df



def theor_vecs_1d(K):
    x = np.cos(2*np.pi*np.arange(K)/K).reshape(-1,1)
    x = x/np.linalg.norm(x)
    y = np.sin(2*np.pi*np.arange(K)/K).reshape(-1,1)
    y = y/np.linalg.norm(y)
    vecs = rotate_eigenvectors(np.hstack((x,y)),samples = [1]*K)
    return vecs


def is_seperated_empirical(l,n,L):
    """A test of whether the empirical eigenvalue is seperated from the bulk

    Args:
        l (float): eigenvalue
        n (int): number of samples (rows)
        L (int): number of independent SNPs (columns)

    Returns:
        bool: Whether the eigenvalue is seperated from the bulk of the Tracy-Widom distribution
    """
    edge_value = (1+np.sqrt(n/L))**2
    return l > edge_value if n<L else np.nan
    
is_seperated_empirical_vectorized = np.vectorize(is_seperated_empirical)

def is_seperated_theoretical(l,n,L):
    """A test of whether the theoretical eigenvalue exceeds the critical value

    Args:
        l (float): eigenvalue
        n (int): number of samples (rows)
        L (int): number of independent SNPs (columns)

    Returns:
        bool: Whether the eigenvalue exceeds the critical value
    """
    if n>L:
        return np.nan
    critical_value = 1+np.sqrt(n/L)
    if l > critical_value:
        return True
    else:
        return False



def preprocess_T(demography,N=1000):
    if "1d" in demography:
        d=1
    elif "2d" in demography:
        d=2
    data_dir = "../../data/" + demography + "/"
    data = np.zeros((1,4))
    folders = os.listdir(data_dir)
    for folder in tqdm(folders):
        K = int(folder[2:])
        data_folders = os.listdir(os.path.join(data_dir,folder))
        for data_folder in data_folders:
            m = float(data_folder[2:])
            if "branch_lengths.pkl" in os.listdir(data_dir+f"K={K}/m={m}"):
                T_list = pd.read_pickle(data_dir+f"K={K}/m={m}/branch_lengths.pkl")/((K**d)*N)
                data_len = len(T_list)
                data_temp = np.hstack((np.array([K**d]*data_len).reshape(-1,1),np.array([m]*data_len).reshape(-1,1),np.arange(1,data_len+1).reshape(-1,1),T_list.reshape(-1,1)))
                data = np.vstack((data,data_temp))
    return data[1:,:-1],data[1:,-1]

def train_regressor_T(demography,N=1000):
    X, y = preprocess_T(demography,N)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    regr = RandomForestRegressor(random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print(f"MSE = {mean_squared_error(y_test, y_pred)}")
    return regr

def theor_eig(demography,regr,param_arr,N=1000):
    if "1d" in demography:
        assert len(param_arr) == 3
        K,m,n = param_arr
        M = m * N
        T_list = [(x * (K - x)) / (2 * M * K) for x in np.arange(1, ((K - 1) / 2) + 1)]
        T_list2 = np.array(T_list+T_list[::-1]) 
        T_list_full = [1]+ list(1+T_list2)
        T_tilde = np.unique(np.real(fft(T_list_full)[1:]))
        return ((1 / regr.predict(np.array([K,m,n]).reshape(1,-1))) * (1 - n * T_tilde)).reshape(1,-1)

def construct_full_eigenvalue_df(demography):
    if "1d" in demography:
        d=1
    else:
        d=2

    input_path = f"../../data/{demography}/"
    folders = os.listdir(input_path)
    if "pop_split" in input_path:
        full_df = construct_partial_eigenvalue_df(input_path,"Fst")
    else:
        dfs = []
        for folder in folders:
            try:
                inner_path = os.path.join(input_path,folder)
                K = int(folder.split(",")[0][2:])
                temp_df = construct_partial_eigenvalue_df(inner_path,"m")
                if "cont" in input_path:
                    N = int(folder.split(",")[1][2:])
                    temp_df.insert(0, 'N', N)
                temp_df.insert(0, 'K', K)
                dfs.append(temp_df)
            except: continue
        full_df = pd.concat(dfs,ignore_index=True)

    pattern = re.compile(r'l_\d+$')
    for column in [s for s in full_df.columns if pattern.match(s)]:
            full_df["is_seperated_"+column] = is_seperated_empirical_vectorized(full_df[[column]].values.reshape(1,-1)[0],(full_df.K*full_df.n).values,full_df.L.values)
            # full_df[f"{col}_theoretical"] = None
    # regr = train_regressor_T(demography)
    
    # K_list = np.unique(full_df.K.values)**d
    # for k in K_list:
    #         data = full_df[full_df.K==k][["K","m","n"]].values
    #         result = np.apply_along_axis(lambda x: theor_eig(demography, regr, x, N=1000), axis=1, arr=data)
    #         full_df[full_df[K ==k]][[f"l_{i}_theor" for i in range(1,int((k-1)/2)+1)]]=result
    return full_df

def find_empirical_cutoff(df,cutoff_param,params,p):
    try:
        if cutoff_param == "m":
            sub_df = df[(df.K==params["K"]) & (df.n==params["n"]) & (df.L==params["L"])]
            inx = np.where(sub_df.loc[:,f"is_seperated_l_{p}"].values)[0][-1]
        elif cutoff_param == "n":
            sub_df = df[(df.K==params["K"]) & (df.m==params["m"]) & (df.L==params["L"])]
            inx = np.where(sub_df.loc[:,f"is_seperated_l_{p}"].values)[0][0]
        elif cutoff_param == "L":
            sub_df = df[(df.K==params["K"]) & (df.m==params["m"]) & (df.n==params["n"])]
            inx = np.where(sub_df.loc[:,f"is_seperated_l_{p}"].values)[0][0]

        return sub_df.loc[:,cutoff_param].values[inx]
    except:
        return np.nan

def find_theoretical_cutoff_single(demography,regr,cutoff_param,cutoff_param_list,params):
    num_trials = len(cutoff_param_list)
    if "d" in demography:
        K = params["K"]
    elif "split" in demography:
        K=2
    elif cutoff_param == "m":
        n=params["n"]
        L=params["L"]
        param_arr = np.hstack((np.ones((num_trials,1))*K,cutoff_param_list.reshape(-1,1),(np.ones((num_trials,1))*n)))
    elif cutoff_param == "Fst":
        n=params["n"]
        L=params["L"]
        param_arr = np.hstack((cutoff_param_list.reshape(-1,1),(np.ones((num_trials,1))*n)))
    elif cutoff_param == "n":
        L=params["L"]
        n=cutoff_param_list
        if "d" in demography:
            m=params["m"]
            param_arr = np.hstack((np.ones((num_trials,1))*K,np.ones((num_trials,1))*m,cutoff_param_list.reshape(-1,1)))
        elif "split" in demography:
            Fst=params["Fst"]
            param_arr = np.hstack((np.ones((num_trials,1))*Fst,cutoff_param_list.reshape(-1,1)))
    elif cutoff_param == "L":
        n=params["n"]
        L=cutoff_param_list
        if "d" in demography:
            m=params["m"]
            param_arr = np.hstack((np.ones((num_trials,1))*K,np.ones((num_trials,1))*m,(np.ones((num_trials,1))*n)))
        elif "split" in demography:
            Fst = params["Fst"]
            param_arr = np.hstack((np.ones((num_trials,1))*Fst,(np.ones((num_trials,1))*n)))

    l = 1+np.sqrt(n/L)
    if isinstance(l,np.ndarray):
        l = l.reshape(-1,1)
    solutions = np.apply_along_axis(lambda x: theor_eig(demography = demography,regr=regr,param_arr=x,N=1000),axis=1,arr=param_arr).reshape(-1,int((K-1)/2))-l
    result_inx = []
    for i in range(solutions.shape[1]):
        try:
            result_inx.append(np.where(solutions[:,i]<0)[0][-1])  
        except:
            result_inx.append(np.nan)
    return solutions,result_inx

def find_theoretical_cutoff(demography,regr,cutoff_param,cutoff_param_list_init,params):
    solutions0, inx0 = find_theoretical_cutoff_single(demography,regr,cutoff_param,cutoff_param_list_init,params) ## the index of an initial Solution
    result = []
    for i in range(len(inx0)): ## Iterating over the cutoffs for all different eigenvalues
        cutoff_param_list = cutoff_param_list_init
        solutions = solutions0
        inx = inx0[i]
        while np.abs(solutions[inx,i]) > 1e-2:
            if np.isnan(inx): ## Case where solution lies below minimum value for the cutoff_list
                if cutoff_param == "m":
                    cutoff_param_list = np.geomspace(cutoff_param_list[0],cutoff_param_list[0]*10,len(cutoff_param_list))[::-1]
                else:
                    cutoff_param_list = np.geomspace(cutoff_param_list[0]/10,cutoff_param_list[0],len(cutoff_param_list))
            elif inx == len(cutoff_param_list)-1: ## Case where solution lies above maximum value for the cutoff_list
                if cutoff_param == "m":
                    cutoff_param_list = np.geomspace(cutoff_param_list[-1]/10,cutoff_param_list[-1],len(cutoff_param_list))[::-1]
                else:
                    cutoff_param_list = np.geomspace(cutoff_param_list[-1],cutoff_param_list[-1]*10,len(cutoff_param_list))
            else:
                if cutoff_param == "m":
                    cutoff_param_list = np.geomspace(cutoff_param_list[inx+1],cutoff_param_list[inx],len(cutoff_param_list))[::-1]
                else:
                    cutoff_param_list = np.geomspace(cutoff_param_list[inx],cutoff_param_list[inx+1],len(cutoff_param_list))


            solutions, inx = find_theoretical_cutoff_single(demography,regr,cutoff_param,cutoff_param_list,params) ## find the index of a new solution
            inx = inx[i]
        result.append(cutoff_param_list[inx])
    return result

def rotate_eigenvectors(eigenvectors, samples, index=0):
    K=len(samples)
    samples_sums = np.cumsum(np.concatenate((np.array([0]),np.array(samples))))

    centroids_vector_1 = [np.mean(eigenvectors[samples_sums[k]:samples_sums[k+1],0]) for k in range(K)]
    centroids_vector_2 = [np.mean(eigenvectors[samples_sums[k]:samples_sums[k+1],1]) for k in range(K)]

    # centroids_vector_1 = [np.mean(eigenvectors[k*n:(k+1)*n,0]) for k in range(K)]
    # centroids_vector_2 = [np.mean(eigenvectors[k*n:(k+1)*n,1]) for k in range(K)]
    centroids_vector = np.vstack((centroids_vector_1,centroids_vector_2)).T ## K*2 vector of centroids

    x1 = np.abs(centroids_vector[index,0])
    x2 = np.abs(centroids_vector[index,1])

    x1_sign = centroids_vector[index,0] > 0
    x2_sign = centroids_vector[index,1] > 0

    if x1_sign and x2_sign: # both positive
        theta = -(np.pi/2-np.arctan(x2/x1))
    elif x1_sign and not x2_sign: # x positive and y negative
        theta = -(np.pi/2+np.arctan(x2/x1))
    elif not x1_sign and x2_sign: # x negative and y postiive
        theta = np.pi/2-np.arctan(x2/x1)
    elif not x1_sign and not x2_sign: # both negative
        theta = np.pi/2+np.arctan(x2/x1)

    Rotation_Matrix = np.array([[np.cos(theta),np.sin(theta)],
                            [-np.sin(theta),np.cos(theta)]])
    Rotated_centroids = np.matmul(Rotation_Matrix,centroids_vector.T).T
    Rotated_eigenvectors = np.matmul(Rotation_Matrix,eigenvectors.T).T
    
    # Reflect if necessary, so that demes are counted clockwise
    if Rotated_centroids[1,0]<0:
        Rotated_centroids[:,0] = -Rotated_centroids[:,0]
        Rotated_eigenvectors[:,0] = -Rotated_eigenvectors[:,0]
    Rotated_eigenvectors = Rotated_eigenvectors/np.linalg.norm(Rotated_eigenvectors,axis=0)

    norm = np.linalg.norm(Rotated_centroids,axis=1)[0]
    Rotated_scaled_eigenvectors = Rotated_eigenvectors/norm
    return Rotated_scaled_eigenvectors

def calculate_theor_vec(demography,param_arr):
    ## param_arr is K and m only
    if demography == "SS_1d":
        K = param_arr[0]
        m = param_arr[1]
        x = np.cos(2*np.pi*np.arange(K)/K).reshape(-1,1)
        x = x/np.linalg.norm(x)
        y = np.sin(2*np.pi*np.arange(K)/K).reshape(-1,1)
        y = y/np.linalg.norm(y)
        vecs = rotate_eigenvectors(np.hstack((x,y)),samples = [1]*K)
        return vecs

def calc_theta(vecs,samples,theor_vec):
    rotated_scaled_vec = rotate_eigenvectors(vecs, samples)
    theor_vec = np.repeat(theor_vec,samples,axis=0)
    rotated_scaled_vec = rotated_scaled_vec/np.linalg.norm(rotated_scaled_vec,axis=0)
    theor_vec = theor_vec/np.linalg.norm(theor_vec,axis=0)
    theta1 = np.arccos(np.dot(rotated_scaled_vec[:,0],theor_vec[:,0]))
    theta2 = np.arccos(np.dot(rotated_scaled_vec[:,1],theor_vec[:,1]))
    return (theta1+theta2)/2

def construct_inconsistency_df(demography):
    demography = "SS_1d"
    input_path = f"../../data/{demography}/"
    folders = os.listdir(input_path)
    data = np.zeros((0,4))

    for folder in folders:
        K = int(folder[2:])
        data_folders = os.listdir(input_path + folder)
        for data_folder in data_folders:
            inner_folder = input_path + folder + "/" + data_folder + "/"
            m=float(data_folder[2:])
            if "eigenvectors.pkl" in os.listdir(inner_folder):
                eigenvector_df = pd.read_pickle(inner_folder + "eigenvectors.pkl")
                n_vec_pairs = int((len(eigenvector_df.columns)-2)/2)
                if data.shape[1] < 4+n_vec_pairs:
                    data = np.hstack((data,np.full((data.shape[0],n_vec_pairs),np.nan)))
                n_list = np.unique(eigenvector_df.n.values).astype(int)[5:25:5]
                L_list = np.unique(eigenvector_df.L.values).astype(int)[::5]
                for n in n_list:
                    for L in L_list:
                        theta_list = []
                        temp_df = eigenvector_df[(eigenvector_df.n == n) & (eigenvector_df.L == L)]
                        if temp_df.shape[0] != 0:
                            for i in range(n_vec_pairs):
                                vecs = temp_df.loc[:,[f"v_{2*i+1}",f"v_{2*i+2}"]].values
                                theta_list.append(calc_theta(vecs,[n]*K,calculate_theor_vec(demography,[K,m])))
                            row = [K,m,n,L] + theta_list
                            row = row + [np.nan] * (data.shape[1]-len(row))
                            data = np.vstack((data,row))
    data_df = pd.DataFrame(data,columns = ["K","m","n","L"] + [f"theta_{i}" for i in range(1,data.shape[1]-4+1)])
    return data_df

def find_empirical_cutoff_vecs(df,cutoff_param,params,p):
    try:
        if cutoff_param == "m":
            sub_df = df[(df.K==params["K"]) & (df.n==params["n"]) & (df.L==params["L"])]
            vec = sub_df.loc[:,f"theta_{p}"].values[::-1]
        elif cutoff_param == "n":
            sub_df = df[(df.K==params["K"]) & (df.m==params["m"]) & (df.L==params["L"])]
            vec = np.where(sub_df.loc[:,f"theta_{p}"].values)
        elif cutoff_param == "L":
            sub_df = df[(df.K==params["K"]) & (df.m==params["m"]) & (df.n==params["n"])]
            vec = np.where(sub_df.loc[:,f"theta_{p}"].values)
        
        vec_bool = vec<0.4
        inx1 = np.where(vec_bool)[0][0]
        vec_bool_rest = vec_bool[inx1:]
        percent_remaining_true = (np.cumsum(vec_bool_rest[::-1])[::-1])/np.arange(1,len(vec_bool_rest)+1)[::-1]
        inx2=np.where(percent_remaining_true>0.95)[0][0]

    except:
        return
        return inx1+inx2

# %%
# inconsistency_df_smaller = construct_inconsistency_df(demography)
n=np.unique(inconsistency_df_smaller.n.values)[0]
L=np.unique(inconsistency_df_smaller.L.values)[0]
K=5
theta_list = inconsistency_df_smaller[(inconsistency_df_smaller.n==n)&(inconsistency_df_smaller.L==L)&(inconsistency_df_smaller.K==5)]
theta_list = theta_list.sort_values("m")
theta_1 = theta_list.theta_1.values
plt.scatter(range(len(theta_1)),theta_1[::-1])
# %%
pattern = re.compile(r'l_\d+$')
cols = [s for s in full_df.columns if pattern.match(s)]
cutoff_m_df = np.zeros((1,3+len(cols)))
cutoff_param = "m"
K_list = np.unique(df.K.values)
n_list = np.unique(df.n.values)
L_list = np.unique(df.L.values)
for K in tqdm(K_list):
    for n in tqdm(n_list):
        for L in L_list:
            params = {"K":K,"n":n,"L":L}
            row = [K,n,L] + [0] * 2*len(cols)
            for i,col in enumerate(cols):
                cutoff = find_empirical_cutoff(df,cutoff_param,params,i+1)
                row[3+i] = cutoff
            if not np.all(np.isnan(row[3:])):
                cutoff_m_df = np.vstack((cutoff_m_df,row))
cutoff_m_df_2 = pd.DataFrame(cutoff_m_df[1:,:],columns=["K","m","n"] + cols)
#%%
demography = "SS_1d"
# regr = train_regressor_T(demography)
cutoff_param = "m"
# params = {"K":9,"m":0.001,"n":100}
params = {"K":9,"n":20,"L":4000}
cutoff_param_list_init = np.linspace(0.001,0.1,25)[::-1]
# cutoff_param_list_init = np.linspace(1e2,1e4,25).astype(int)
#%%
print(find_theoretical_cutoff(demography,regr,cutoff_param,cutoff_param_list_init,params))

#%%
L=6816
inconsistency_df_smaller_5 = inconsistency_df_smaller[inconsistency_df_smaller.K==5]
inconsistency_df_smaller_5=inconsistency_df_smaller_5[inconsistency_df_smaller_5.L==L]
inconsistency_df_smaller_5 = inconsistency_df_smaller_5.loc[:,["m","n","theta_1"]]
inconsistency_df_smaller_5 = inconsistency_df_smaller_5.sort_values(["n","m"])
x = inconsistency_df_smaller_5.pivot(index="m",columns="n",values="theta_1")
fig,ax = plt.subplots(1,2,figsize = (12,5))
sns.heatmap(x,ax=ax[0])
sns.heatmap(x>0.4,ax=ax[1])
# %%
array([1200., 1698., 2403., 3402., 4815., 6816.])
#%%
